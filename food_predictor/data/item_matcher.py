import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging
import pandas as pd 
class FoodItemMatcher:
    """Two-tier hash table system for efficient food item matching."""

    def __init__(self, item_metadata=None):
        self.direct_lookup = {}  # Tier 1: Direct lookup hash table
        self.token_to_items = {}  # Tier 2: Token-based lookup
        self.category_items = {}  # Items organized by category

        if item_metadata:
            self.build_hash_tables(item_metadata)

    def build_hash_tables(self, item_metadata):
        """Build the two-tier hash table system from item metadata."""
        # Reset tables
        self.direct_lookup = {}
        self.token_to_items = {}
        self.category_items = {}

        # Process each item
        for item_name, metadata in item_metadata.items():
            if not item_name or pd.isna(item_name):
                continue
            if isinstance(item_name,tuple) and len(item_name) > 0:
                item_name = item_name[0]

            # Standardize name
            std_name = self._standardize_name(item_name)
            if 'sambar' in std_name.lower():
                print(f"DEBUG: Adding '{std_name}' to direct lookup from '{item_name}'")
            # Add to direct lookup (Tier 1)
            self.direct_lookup[std_name] = item_name

            # Add common variants
            variants = self._generate_variants(std_name)
            for variant in variants:
                if variant not in self.direct_lookup:
                    self.direct_lookup[variant] = item_name

            # Add to token lookup (Tier 2)
            tokens = self._tokenize(std_name)
            for token in tokens:
                if token not in self.token_to_items:
                    self.token_to_items[token] = []
                if item_name not in self.token_to_items[token]:
                    self.token_to_items[token].append(item_name)

            # Add to category items
            category = metadata.category
            if category not in self.category_items:
                self.category_items[category] = []
            self.category_items[category].append(item_name)

    def find_item(self, query_item, item_metadata):
        """Find a food item using the two-tier hash table approach."""
        if not query_item or pd.isna(query_item):
            return None, None

        # Standardize query - handle '>' prefix automatically
        std_query = self._standardize_name(query_item)
        std_query = std_query.replace('> ', '').replace('>', '')  # Remove '>' prefixes

        # Tier 1: Try direct lookup first (fast path)
        if std_query in self.direct_lookup:
            item_name = self.direct_lookup[std_query]
            if item_name in item_metadata:
                return item_name, item_metadata[item_name]

        # Additional direct lookups for common variations
        # Try with spaces removed
        compact_query = std_query.replace(' ', '')
        for key in self.direct_lookup:
            # Check if key is a tuple and extract the string part if it is
            if isinstance(key, tuple):
                key_str = key[0] if len(key) > 0 else ""
            else:
                key_str = key
                
            compact_key = key_str.replace(' ', '')
            if compact_query == compact_key:
                item_name = self.direct_lookup[key]
                if item_name in item_metadata:
                    return item_name, item_metadata[item_name]

        # Tier 2: Enhanced token-based lookup
        tokens = self._tokenize(std_query)
        if tokens:
            # Find candidates with improved scoring
            candidates = {}
            for token in tokens:
                # Handle token variations (plurals, singulars)
                token_variants = [token]
                if token.endswith('s'):
                    token_variants.append(token[:-1])  # Remove 's' for plurals
                elif len(token) > 3:
                    token_variants.append(token + 's')  # Add 's' for singulars

                for variant in token_variants:
                    if variant in self.token_to_items:
                        for item_name in self.token_to_items[variant]:
                            if item_name not in candidates:
                                candidates[item_name] = 0
                            candidates[item_name] += 1

            # Enhance scoring with additional contextual factors
            scored_candidates = []
            for item_name, token_matches in candidates.items():
                if item_name in item_metadata:
                    item_tokens = self._tokenize(self._standardize_name(item_name))
                    if not item_tokens:
                        continue

                    # Basic token match score
                    token_score = token_matches / max(len(tokens), len(item_tokens))

                    # Enhanced substring matching
                    contains_score = 0
                    if std_query in self._standardize_name(item_name):
                        contains_score = 0.8
                    elif self._standardize_name(item_name) in std_query:
                        contains_score = 0.6

                    # Word overlap score (considering word position)
                    word_overlap = 0
                    std_query_words = std_query.split()
                    item_words = self._standardize_name(item_name).split()
                    for i, qword in enumerate(std_query_words):
                        for j, iword in enumerate(item_words):
                            if qword == iword:
                                # Words in same position get higher score
                                pos_factor = 1.0 - 0.1 * abs(i - j)
                                word_overlap += pos_factor

                    if len(std_query_words) > 0:
                        word_overlap_score = word_overlap / len(std_query_words)
                    else:
                        word_overlap_score = 0

                    # Combined score with weights
                    final_score = max(token_score * 0.4 + contains_score * 0.4 + word_overlap_score * 0.2,
                                    contains_score)

                    scored_candidates.append((item_name, final_score))

            # Sort by score and get best match
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = scored_candidates[0]

                # Lower threshold for matching (0.4 instead of 0.5)
                if best_match[1] >= 0.4:
                    return best_match[0], item_metadata[best_match[0]]

        return None, None

    def _standardize_name(self, name):
        """Standardize item name for matching."""
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = " ".join(name.split())  # Normalize whitespace
        return name

    def _tokenize(self, text):
        """Split text into tokens for matching."""
        if not text:
            return []

        # Simple tokenization by whitespace
        tokens = text.split()

        # Remove very common words and short tokens
        stop_words = {"and", "with", "the", "in", "of", "a", "an"}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    def _generate_variants(self, name):
        """Generate common variants of a food item name."""
        variants = []

        # Common misspellings and variations
        replacements = {
                "biryani": ["biriyani", "briyani", "biryani"],
                "chicken": ["chiken", "chikken", "checken"],
                "paneer": ["panner", "panir", "pnr"],
                "masala": ["masala", "masaala", "masalla"]
            }

            # Generate simple word order variations
        words = name.split()
        if len(words) > 1:
            # Add reversed order for two-word items
            variants.append(" ".join(reversed(words)))

            # Apply common spelling variations
        for word, alternatives in replacements.items():
            if word in name:
                for alt in alternatives:
                    variant = name.replace(word, alt)
                    variants.append(variant)

        return variants