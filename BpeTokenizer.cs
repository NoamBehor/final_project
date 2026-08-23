using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace RadioNLP.TokenizerCore
{
      /// Dual-representation output: Text tokens for BM25 and Numeric IDs for Embeddings.
    public class TokenizationResult
    {
        public List<string> Tokens { get; set; } = new List<string>();
        public List<int> TokenIds { get; set; } = new List<int>();
    }

    /// Schema used to serialize and deserialize the trained BPE model.
    public class TokenizerModelData
    {
        public List<string> ProtectedTerms { get; set; } = new List<string>();
        public List<string[]> MergeRules { get; set; } = new List<string[]>();
        public Dictionary<string, int> TokenToId { get; set; } = new Dictionary<string, int>();
        public bool NormalizeFinalLetters { get; set; }
    }

    /// Byte Pair Encoding (BPE) Tokenizer for Hebrew radio and military communications.
    public class BpeTokenizer
    {
        // Special Tokens
        public const string PadToken = "[PAD]";
        public const string UnkToken = "[UNK]";
        public const string ClsToken = "[CLS]";
        public const string SepToken = "[SEP]";
        public const string WordEndSymbol = "</w>";

        public const int PadId = 0;
        public const int UnkId = 1;
        public const int ClsId = 2;
        public const int SepId = 3;

        // Internal State Mappings
        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<int, string> _invVocab;
        private readonly HashSet<string> _protectedTerms;
        private readonly List<(string First, string Second)> _mergeRules;
        private readonly bool _normalizeFinalLetters;

  