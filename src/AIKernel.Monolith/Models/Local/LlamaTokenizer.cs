using System.Globalization;
using System.Text;

namespace AIKernel.Monolith.Models.Local;

internal sealed class LlamaTokenizer
{
    internal const int Begin = 1;
    internal const int End = 2;

    private readonly string[] pieces;
    private readonly float[] scores;
    private readonly byte?[] rawBytes;
    private readonly int[] byteTokens = new int[256];
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);

    internal LlamaTokenizer(string path, int count)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        var maxLength = reader.ReadInt32();
        if (maxLength is < 1 or > 4096)
        {
            throw new InvalidDataException("Invalid tokenizer token length.");
        }

        pieces = new string[count];
        scores = new float[count];
        rawBytes = new byte?[count];
        Array.Fill(byteTokens, 0); // Unknown token when the vocabulary has no byte fallback.
        var utf8 = new UTF8Encoding(false, true);
        for (var id = 0; id < count; id++)
        {
            scores[id] = reader.ReadSingle();
            var length = reader.ReadInt32();
            if (!float.IsFinite(scores[id]) || length is < 1 || length > maxLength)
            {
                throw new InvalidDataException("Invalid tokenizer entry.");
            }

            var bytes = reader.ReadBytes(length);
            if (bytes.Length != length)
            {
                throw new EndOfStreamException("Incomplete tokenizer entry.");
            }

            var piece = utf8.GetString(bytes);
            pieces[id] = piece;
            if (id > End)
            {
                vocabulary.TryAdd(piece, id);
                if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>' &&
                    byte.TryParse(piece.AsSpan(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var value))
                {
                    rawBytes[id] = value;
                    byteTokens[value] = id;
                }
            }
        }

        if (reader.BaseStream.Position != reader.BaseStream.Length)
        {
            throw new InvalidDataException("The tokenizer vocabulary does not match the checkpoint.");
        }
    }

    internal List<int> Encode(string prompt, CancellationToken token)
    {
        var tokens = new List<int> { Begin };
        Span<byte> bytes = stackalloc byte[4];
        foreach (var rune in (" " + prompt).EnumerateRunes())
        {
            if (vocabulary.TryGetValue(rune.ToString(), out var id))
            {
                tokens.Add(id);
            }
            else
            {
                var count = rune.EncodeToUtf8(bytes);
                for (var i = 0; i < count; i++)
                {
                    tokens.Add(byteTokens[bytes[i]]);
                }
            }
        }

        while (true)
        {
            token.ThrowIfCancellationRequested();
            var best = -1;
            var merged = 0;
            var score = float.NegativeInfinity;
            for (var i = 1; i + 1 < tokens.Count; i++)
            {
                if (vocabulary.TryGetValue(pieces[tokens[i]] + pieces[tokens[i + 1]], out var id) && scores[id] > score)
                {
                    best = i;
                    merged = id;
                    score = scores[id];
                }
            }

            if (best < 0)
            {
                return tokens;
            }

            tokens[best] = merged;
            tokens.RemoveAt(best + 1);
        }
    }

    internal string Decode(int token, Decoder decoder)
    {
        if (rawBytes[token] is not { } value)
        {
            return Flush(decoder) + (token == 0 ? string.Empty : pieces[token]);
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
}
