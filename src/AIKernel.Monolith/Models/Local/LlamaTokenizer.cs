using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace AIKernel.Monolith.Models.Local;

internal sealed class LlamaTokenizer
{
    internal const int Begin = 1;
    internal const int End = 2;
    private static readonly UTF8Encoding Utf8 = new(false, true);

    private readonly string[] pieces;
    private readonly byte?[] rawBytes;
    private readonly int[] byteTokens = new int[256];
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(int, int), (int Id, float Score)> merges = [];

    internal LlamaTokenizer(string path, int count) : this(Read(path, count), count) { }

    private LlamaTokenizer(Vocabulary data, int count)
    {
        if (data.Pieces.Length != count || data.Scores.Length != count)
        {
            throw new InvalidDataException("The tokenizer vocabulary does not match the checkpoint.");
        }

        pieces = data.Pieces;
        rawBytes = new byte?[count];
        for (var id = 0; id < count; id++)
        {
            var piece = pieces[id]?.Replace('▁', ' ') ?? throw new InvalidDataException("Missing tokenizer ID.");
            if (piece.Length is < 1 or > 4096 || !float.IsFinite(data.Scores[id]))
            {
                throw new InvalidDataException("Invalid tokenizer entry.");
            }

            pieces[id] = piece;
            if (id <= End) continue;
            vocabulary.TryAdd(piece, id);
            if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>' &&
                byte.TryParse(piece.AsSpan(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var value))
            {
                rawBytes[id] = value;
                byteTokens[value] = id;
            }
        }

        var lookup = vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();
        if (data.MergeRules is { } rules)
        {
            for (var rank = 0; rank < rules.Length; rank++)
            {
                var (left, right) = rules[rank];
                left = left.Replace('▁', ' ');
                right = right.Replace('▁', ' ');
                if (vocabulary.TryGetValue(left, out var a) && vocabulary.TryGetValue(right, out var b) &&
                    vocabulary.TryGetValue(left + right, out var id))
                {
                    merges.TryAdd((a, b), (id, -rank));
                }
            }
        }
        else
        {
            // Derive SentencePiece BPE candidate pairs once. The merge loop uses integer keys.
            foreach (var (piece, id) in vocabulary)
            {
                for (var split = 1; split < piece.Length; split++)
                {
                    if (lookup.TryGetValue(piece.AsSpan(0, split), out var a) &&
                        lookup.TryGetValue(piece.AsSpan(split), out var b))
                    {
                        merges.TryAdd((a, b), (id, data.Scores[id]));
                    }
                }
            }
        }
    }

    internal static LlamaTokenizer FromEmbedded(MappedModelFile? file, int count)
        => file?.Tokens is { } tokens && file.Scores is { } scores
            ? new LlamaTokenizer(new Vocabulary(tokens, scores), count)
            : throw new UnsupportedModelException("Provide a matching tokenizer for this model.");

    internal List<int> Encode(string prompt, CancellationToken token)
    {
        var tokens = new List<int>(prompt.Length + 2) { Begin };
        if (vocabulary.TryGetValue(" ", out var prefix)) tokens.Add(prefix);
        var lookup = vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();
        var remaining = prompt.AsSpan();
        Span<byte> bytes = stackalloc byte[4];
        while (!remaining.IsEmpty)
        {
            var status = Rune.DecodeFromUtf16(remaining, out var rune, out var consumed);
            if (status != OperationStatus.Done) consumed = 1;
            if (lookup.TryGetValue(remaining[..consumed], out var id))
            {
                tokens.Add(id);
            }
            else
            {
                var count = rune.EncodeToUtf8(bytes);
                for (var i = 0; i < count; i++) tokens.Add(byteTokens[bytes[i]]);
            }

            remaining = remaining[consumed..];
        }

        while (true)
        {
            token.ThrowIfCancellationRequested();
            var best = -1;
            var merged = 0;
            var score = float.NegativeInfinity;
            for (var i = 1; i + 1 < tokens.Count; i++)
            {
                if (merges.TryGetValue((tokens[i], tokens[i + 1]), out var candidate) && candidate.Score > score)
                {
                    best = i;
                    merged = candidate.Id;
                    score = candidate.Score;
                }
            }

            if (best < 0) return tokens;
            tokens[best] = merged;
            tokens.RemoveAt(best + 1);
        }
    }

    internal string Decode(int token, Decoder decoder)
    {
        if (rawBytes[token] is not { } value)
        {
            var prefix = Flush(decoder);
            return token == 0 ? prefix : prefix + pieces[token];
        }

        Span<char> chars = stackalloc char[2];
        var length = decoder.GetChars([value], chars, flush: false);
        return new string(chars[..length]);
    }

    internal static string Flush(Decoder decoder)
    {
        Span<char> chars = stackalloc char[2];
        var length = decoder.GetChars([], chars, flush: true);
        return new string(chars[..length]);
    }

    private static Vocabulary Read(string path, int count)
    {
        using var file = File.OpenRead(path);
        if (file.Length > 64 * 1024 * 1024) throw new InvalidDataException("Tokenizer exceeds 64 MiB.");
        var pieces = new string[count];
        var scores = new float[count];
        if (Path.GetExtension(path).Equals(".json", StringComparison.OrdinalIgnoreCase))
        {
            using var document = JsonDocument.Parse(file);
            var root = document.RootElement;
            var model = root.GetProperty("model");
            if (model.GetProperty("type").GetString() != "BPE" ||
                root.TryGetProperty("pre_tokenizer", out var pre) && pre.GetRawText().Contains("ByteLevel", StringComparison.Ordinal))
                throw new UnsupportedModelException("Expected a SentencePiece-style BPE tokenizer.json.");

            foreach (var piece in model.GetProperty("vocab").EnumerateObject())
            {
                var id = piece.Value.GetInt32();
                if ((uint)id >= (uint)count || pieces[id] is not null)
                    throw new InvalidDataException("Invalid or repeated tokenizer ID.");
                pieces[id] = piece.Name;
            }

            var rules = model.GetProperty("merges").EnumerateArray().Select(rule =>
            {
                if (rule.ValueKind == JsonValueKind.Array)
                {
                    if (rule.GetArrayLength() != 2) throw new InvalidDataException("Invalid BPE merge.");
                    return (rule[0].GetString()!, rule[1].GetString()!);
                }

                var text = rule.GetString()!;
                var separator = text.IndexOf(' ');
                return separator > 0 ? (text[..separator], text[(separator + 1)..]) :
                    throw new InvalidDataException("Invalid BPE merge.");
            }).ToArray();
            return new(pieces, scores, rules);
        }

        if (Path.GetExtension(path).Equals(".model", StringComparison.OrdinalIgnoreCase))
        {
            var data = new byte[checked((int)file.Length)];
            file.ReadExactly(data);
            var proto = new ProtoReader(data);
            var id = 0;
            var isBpe = false;
            while (proto.Next(out var field, out var wire))
            {
                if (field == 2 && wire == 2)
                {
                    isBpe = IsSentencePieceBpe(proto.Bytes());
                    continue;
                }

                if (field != 1 || wire != 2)
                {
                    proto.Skip(wire);
                    continue;
                }

                if (id == count) throw new InvalidDataException("Extra SentencePiece vocabulary entries.");
                var entry = new ProtoReader(proto.Bytes());
                while (entry.Next(out var tag, out var kind))
                {
                    if (tag == 1 && kind == 2) pieces[id] = Utf8.GetString(entry.Bytes());
                    else if (tag == 2 && kind == 5) scores[id] = BinaryPrimitives.ReadSingleLittleEndian(entry.Take(4));
                    else entry.Skip(kind);
                }

                id++;
            }

            if (!isBpe) throw new UnsupportedModelException("Expected a SentencePiece BPE model with Llama token IDs.");
            if (id != count) throw new InvalidDataException("Incomplete SentencePiece vocabulary.");
            return new(pieces, scores);
        }

        using var reader = new BinaryReader(file);
        var maximum = reader.ReadInt32();
        if (maximum is < 1 or > 4096) throw new InvalidDataException("Invalid tokenizer token length.");
        for (var id = 0; id < count; id++)
        {
            scores[id] = reader.ReadSingle();
            var length = reader.ReadInt32();
            if (length is < 1 || length > maximum) throw new InvalidDataException("Invalid tokenizer entry.");
            var bytes = reader.ReadBytes(length);
            if (bytes.Length != length) throw new EndOfStreamException();
            pieces[id] = Utf8.GetString(bytes);
        }

        if (file.Position != file.Length) throw new InvalidDataException("Extra tokenizer entries.");
        return new(pieces, scores);
    }

    private static bool IsSentencePieceBpe(ReadOnlySpan<byte> data)
    {
        var trainer = new ProtoReader(data);
        ulong type = 1, unk = 0, bos = Begin, eos = End;
        while (trainer.Next(out var tag, out var wire))
        {
            if (wire != 0)
            {
                trainer.Skip(wire);
                continue;
            }

            var value = trainer.Varint();
            switch (tag)
            {
                case 3: type = value; break;
                case 40: unk = value; break;
                case 41: bos = value; break;
                case 42: eos = value; break;
            }
        }

        return type == 2 && unk == 0 && bos == Begin && eos == End;
    }

    private sealed record Vocabulary(string[] Pieces, float[] Scores, (string Left, string Right)[]? MergeRules = null);

    private ref struct ProtoReader(ReadOnlySpan<byte> data)
    {
        private ReadOnlySpan<byte> remaining = data;

        internal bool Next(out int field, out int wire)
        {
            if (remaining.IsEmpty)
            {
                field = wire = 0;
                return false;
            }

            var key = checked((uint)Varint());
            field = (int)(key >> 3);
            wire = (int)(key & 7);
            if (field == 0) throw new InvalidDataException("Invalid SentencePiece field.");
            return true;
        }

        internal ReadOnlySpan<byte> Take(int count)
        {
            if ((uint)count > (uint)remaining.Length) throw new EndOfStreamException();
            var value = remaining[..count];
            remaining = remaining[count..];
            return value;
        }

        internal ReadOnlySpan<byte> Bytes() => Take(checked((int)Varint()));

        internal ulong Varint()
        {
            ulong value = 0;
            for (var shift = 0; shift < 64; shift += 7)
            {
                var b = Take(1)[0];
                if (shift == 63 && b > 1) break;
                value |= (ulong)(b & 127) << shift;
                if (b < 128) return value;
            }

            throw new InvalidDataException("Invalid SentencePiece integer.");
        }

        internal void Skip(int wire)
        {
            switch (wire)
            {
                case 0: _ = Varint(); break;
                case 1: _ = Take(8); break;
                case 2: _ = Bytes(); break;
                case 5: _ = Take(4); break;
                default: throw new InvalidDataException("Unsupported SentencePiece field.");
            }
        }
    }
}
