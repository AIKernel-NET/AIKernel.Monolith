using System.Text.RegularExpressions;

namespace AIKernel.Monolith.Rag;

internal sealed partial class RagKeywordQuery
{
    // Keep identifiers intact even when they are adjacent to Japanese text.
    private const string Japanese = @"\p{IsCJKUnifiedIdeographs}\p{IsHiragana}\p{IsKatakana}";
    private const string Word = @"[\p{L}\p{M}\p{Nd}-[" + Japanese + "]]+";
    private readonly HashSet<string> terms;
    private readonly HashSet<string> identifiers;

    internal RagKeywordQuery(string query)
    {
        terms = Tokens().Matches(query).Select(match => match.Value).ToHashSet(StringComparer.OrdinalIgnoreCase);
        identifiers = terms.Where(term => term.Any(char.IsLetter) && term.Any(char.IsDigit))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    internal double Combine(RagChunk chunk, double cosine)
    {
        if (terms.Count == 0)
        {
            return cosine;
        }

        var matches = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        AddMatches(chunk.Source.DocumentId);
        AddMatches(chunk.Source.Title);
        AddMatches(chunk.Text);

        var coverage = (double)matches.Count / terms.Count;
        var score = (cosine + 2 * coverage) / 3;
        // Reserve the upper half of the score range for exact identifier matches.
        return identifiers.Count == 0 ? score : (score + (matches.Overlaps(identifiers) ? 1 : 0)) / 2;

        void AddMatches(string? text)
        {
            if (text is null)
            {
                return;
            }

            foreach (Match token in Tokens().Matches(text))
            {
                if (terms.Contains(token.Value))
                {
                    matches.Add(token.Value);
                }
            }
        }
    }

    [GeneratedRegex(Word + "(?:[-_./]" + Word + ")*|[" + Japanese + "]+", RegexOptions.CultureInvariant)]
    private static partial Regex Tokens();
}
