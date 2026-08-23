using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Search.Engine.Core
{
    // Simple data model for search results
    public class SearchResult
    {
        public string DocId { get; set; }
        public double Score { get; set; }

        public SearchResult(string docId, double score)
        {
            DocId = docId;
            Score = score;
        }
    }

    public class BM25
    {
        // Global constants for configuration
        private const string DEFAULT_STOPWORDS_PATH = "hebrew_stopwords.txt";
        private const string HEBREW_AND_NUMBERS_PATTERN = @"[^\u0590-\u05FF0-9\s]";

        // Algorithm hyperparameters
        private readonly double _k1;
        private readonly double _b;

        // In-memory indexing structures
        private readonly HashSet<string> _stopwords;
        private readonly Dictionary<string, int> _docLengths;
        private readonly Dictionary<string, Dictionary<string, int>> _invertedIndex;

        private int _corpusSize;
        private double _avgdl;

        // Constructor accepting algorithm parameters with clean defaults
        public BM25(string stopwordsFilePath = DEFAULT_STOPWORDS_PATH, double k1 = 1.5, double b = 0.75)
        {
            _k1 = k1;
            _b = b;
            _stopwords = LoadHebrewStopwords(stopwordsFilePath);

            _docLengths = new Dictionary<string, int>();
            _invertedIndex = new Dictionary<string, Dictionary<string, int>>();
            _corpusSize = 0;
            _avgdl = 0.0;
        }

        // Load stopwords from file with UTF-8 encoding
        private HashSet<string> LoadHebrewStopwords(string filePath)
        {
            var set = new HashSet<string>();

            if (!File.Exists(filePath))
            {
                Console.WriteLine($"[Warning] Stopwords file not found at '{filePath}'");
                return set;
            }

            try
            {
                string[] lines = File.ReadAllLines(filePath, Encoding.UTF8);
                foreach (var line in lines)
                {
                    string word = line.Trim();
                    if (!string.IsNullOrEmpty(word))
                    {
                        set.Add(word);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Error] Could not load stopwords: {ex.Message}");
            }

            return set;
        }

        // Clean text preserving Hebrew letters and digits
        public List<string> PreprocessHebrew(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new List<string>();
            }

            // Using the defined constant pattern
            string cleaned = Regex.Replace(text, HEBREW_AND_NUMBERS_PATTERN, "");

            // Split by whitespace
            string[] words = cleaned.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var tokens = new List<string>();

            foreach (var word in words)
            {
                if (!_stopwords.Contains(word))
                {
                    tokens.Add(word);
                }
            }

            return tokens;
        }

        // Build inverted index in RAM from corpus dictionary
        public void Fit(Dictionary<string, string> corpus)
        {
            _corpusSize = corpus.Count;
            if (_corpusSize == 0)
            {
                Console.WriteLine("[Warning] Corpus is empty.");
                return;
            }

            int totalLength = 0;

            foreach (var doc in corpus)
            {
                string docId = doc.Key;
                List<string> tokens = PreprocessHebrew(doc.Value);

                int docLen = tokens.Count;
                _docLengths[docId] = docLen;
                totalLength += docLen;

                foreach (var token in tokens)
                {
                    if (!_invertedIndex.ContainsKey(token))
                    {
                        _invertedIndex[token] = new Dictionary<string, int>();
                    }

                    if (!_invertedIndex[token].ContainsKey(docId))
                    {
                        _invertedIndex[token][docId] = 1;
                    }
                    else
                    {
                        _invertedIndex[token][docId]++;
                    }
                }
            }

            _avgdl = (double)totalLength / _corpusSize;
        }

        // Calculate IDF score for a token
        private double CalcIdf(string word)
        {
            if (!_invertedIndex.ContainsKey(word))
            {
                return 0.0;
            }

            int docFreq = _invertedIndex[word].Count;
            return Math.Log((_corpusSize - docFreq + 0.5) / (docFreq + 0.5) + 1.0);
        }

        // Search query and rank top documents
        public List<SearchResult> Search(string query, int topK = 3)
        {
            List<string> queryTokens = PreprocessHebrew(query);
            var scores = new Dictionary<string, double>();

            foreach (var token in queryTokens)
            {
                if (!_invertedIndex.ContainsKey(token))
                {
                    continue;
                }

                double idf = CalcIdf(token);

                foreach (var entry in _invertedIndex[token])
                {
                    string docId = entry.Key;
                    int tf = entry.Value;
                    int docLen = _docLengths[docId];

                    double num = tf * (_k1 + 1);
                    double den = tf + _k1 * (1 - _b + _b * (docLen / _avgdl));
                    double score = idf * (num / den);

                    if (!scores.ContainsKey(docId))
                    {
                        scores[docId] = 0;
                    }

                    scores[docId] += score;
                }
            }

            return scores.Select(x => new SearchResult(x.Key, x.Value))
                         .OrderByDescending(r => r.Score)
                         .Take(topK)
                         .ToList();
        }
    }
}