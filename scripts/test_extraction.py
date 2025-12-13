"""Test document extraction and output results for manual review."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.services.document_processor import DocumentProcessor
from backend.config import settings

# Mock services for standalone testing
class MockEmbeddingService:
    async def embed_texts(self, texts):
        return [[0.1] * 1536 for _ in texts]

class MockChatService:
    def __init__(self):
        import re
        from collections import Counter

    async def generate_completion(self, prompt, max_tokens=500, temperature=0.7):
        # Match production prompts exactly
        if "concise 2-3 sentence summary" in prompt.lower():
            return self._generate_summary_from_prompt(prompt)
        elif "key terms/keywords" in prompt.lower():
            keywords = self._extract_keywords_from_prompt(prompt)
            return ", ".join(keywords)
        else:
            return "Test response for unknown prompt type"

    def _generate_summary_from_prompt(self, prompt):
        """Generate a concise 2-3 sentence summary from the document content."""
        import re

        # Extract the document content from the prompt
        content_match = re.search(r'Document content.*?:\s*(.+?)(?=Focus on|$)', prompt, re.DOTALL)
        if not content_match:
            return "Summary of research document with key findings and analysis."

        content = content_match.group(1).strip()

        # Extract title
        title_match = re.search(r'titled\s*["\'](.*?)["\']', prompt)
        title = title_match.group(1) if title_match else "research document"

        # Clean and normalize content
        clean_content = self._clean_document_content(content)

        # Extract main topic/theme from the document
        main_topics = self._extract_main_topics(clean_content, title)

        # Generate a descriptive summary based on content analysis
        summary = self._generate_descriptive_summary(clean_content, title, main_topics)

        return summary
    
    def _extract_main_topics(self, content, title):
        """Extract main topics/themes from the document."""
        import re
        from collections import Counter

        # Look for repeated important phrases (2-4 words)
        # Lowercase for matching
        content_lower = content.lower()

        # Extract multi-word phrases
        phrases = re.findall(r'\b[a-z]+(?:\s+[a-z]+){1,3}\b', content_lower)

        # Filter and count
        phrase_counts = Counter()
        stop_phrases = {'the study', 'this study', 'the research', 'this research', 'we found',
                       'the results', 'our findings', 'the data', 'our study', 'we analyzed'}

        for phrase in phrases:
            # Skip very common phrases and those with too many stop words
            if phrase not in stop_phrases and len(phrase) > 5:
                phrase_counts[phrase] += 1

        # Get top topics (minimum 2 occurrences)
        topics = [phrase for phrase, count in phrase_counts.most_common(10) if count >= 2]

        return topics[:5] if topics else []

    def _generate_descriptive_summary(self, content, title, main_topics):
        """Generate a descriptive summary based on content analysis."""
        import re

        # Extract sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        # Filter out references, headers, author names, and title-like content
        content_sentences = []
        for sent in sentences[:30]:  # Check more sentences
            # Skip if it's a reference
            if self._is_reference_line(sent):
                continue

            # Skip if it looks like a title/header (short and doesn't end with proper punctuation)
            if len(sent.split()) < 8 and not sent.rstrip().endswith(('.', '!', '?')):
                continue

            # Skip if it contains multiple capitalized names (likely author list)
            cap_words = re.findall(r'\b[A-Z][a-z]+\b', sent)
            if len(cap_words) > 4:  # Too many capitalized words, likely names
                continue

            # Skip if it starts with title-like patterns
            if re.match(r'^(What|Who|When|Where|Why|How)\s+\w+\s+\w+\s*$', sent, re.IGNORECASE):
                continue

            content_sentences.append(sent)
            if len(content_sentences) >= 8:
                break

        # Try to find defining or explanatory sentences
        summary_candidates = []
        for sent in content_sentences:
            sent_lower = sent.lower()
            # Look for definitional, explanatory, or substantive sentences
            if any(phrase in sent_lower for phrase in ['is an', 'is the', 'refers to', 'defined as',
                                                       'aims to', 'focuses on', 'examines', 'presents',
                                                       'analyzes', 'explores', 'investigates', 'study of',
                                                       'research on', 'analysis of', 'provides', 'includes',
                                                       'demonstrates', 'shows that', 'found that', 'compared']):
                # Additional check: sentence should have some substantive content
                if len(sent.split()) >= 8:
                    summary_candidates.append(sent)

        # Use the first good summary sentence
        if summary_candidates:
            summary = summary_candidates[0]
            # Truncate if too long
            if len(summary) > 250:
                summary = summary[:247] + "..."
            return summary

        # If no definitional sentences, construct from main topics
        if main_topics and len(main_topics) >= 2:
            topics_str = ", ".join(main_topics[:3])
            return f"This document discusses {topics_str} with research findings and analysis."

        # Use first meaningful content sentence (skip very short ones)
        substantive_sentences = [s for s in content_sentences if len(s.split()) >= 10]
        if substantive_sentences:
            summary = substantive_sentences[0]
            if len(summary) > 250:
                summary = summary[:247] + "..."
            return summary

        return f"Research document analyzing {title.replace('_', ' ').replace('-', ' ').lower()} with clinical and methodological findings."

    def _clean_document_content(self, content):
        """Clean document content by removing headers, footers, slide indicators, etc."""
        import re

        # Remove header/footer patterns
        content = re.sub(r'\d*Insert\s*>\s*Header\s*&\s*footer.*?\d{2}-[A-Za-z]{3}-\d{2}', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Insert\s*>\s*Header\s*&\s*footer', '', content, flags=re.IGNORECASE)

        # Remove slide/page indicators
        content = re.sub(r'\bSlide\s+\d+\b', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\bPage\s+\d+\b', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\b\d+\s*(?:of|/)\s*\d+\b', '', content)

        # Remove standalone date patterns (likely headers)
        content = re.sub(r'^\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$',
                        '', content, flags=re.MULTILINE | re.IGNORECASE)

        # Remove standalone numbers (likely page numbers)
        content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        return content
    
    def _is_reference_line(self, line):
        """Check if a line is likely a reference or citation."""
        import re

        # Check for common reference patterns
        reference_patterns = [
            r'\bdoi:\s*\S+',  # DOI
            r'\bPMID:\s*\d+',  # PubMed ID
            r'et\s+al\.',  # Citation style
            r'\b\d{4}\s*;',  # Year with semicolon (journal style)
            r'^\[?\d+\]',  # Numbered reference
        ]

        for pattern in reference_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False
    
    def _extract_keywords_from_prompt(self, prompt):
        """Extract meaningful keywords from the document content."""
        import re
        from collections import Counter

        # Extract the document content from the prompt
        content_match = re.search(r'Document content.*?:\s*(.+?)(?=Provide only|Focus on|$)', prompt, re.DOTALL)
        if not content_match:
            return ["research", "clinical study", "medical data", "health outcomes", "data analysis"]

        content = content_match.group(1).strip()

        # Clean content - remove references and headers
        clean_content = self._clean_document_content(content)

        # Remove reference lines
        lines = clean_content.split('.')
        filtered_lines = []
        for line in lines:
            if not self._is_reference_line(line):
                filtered_lines.append(line)
        clean_content = '. '.join(filtered_lines)

        # Extract keywords using multiple strategies
        keywords = []

        # Strategy 1: Acronyms and abbreviations (likely important terms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        acronym_counts = Counter(acronyms)
        # Keep acronyms that appear at least twice and are likely meaningful
        # Filter out reference identifiers
        excluded_acronyms = {'THE', 'AND', 'FOR', 'ARE', 'PMID', 'DOI', 'ISBN', 'URL', 'PDF', 'HTML'}
        meaningful_acronyms = [acr for acr, count in acronym_counts.items()
                              if count >= 2 and len(acr) >= 2 and acr not in excluded_acronyms]
        keywords.extend(meaningful_acronyms[:3])

        # Strategy 2: Capitalized multi-word terms (proper nouns, technical terms)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        # Filter out author names and dates
        filtered_cap_phrases = []
        for phrase in cap_phrases:
            # Skip if it looks like a date or contains common author name patterns
            if not re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})\b', phrase):
                filtered_cap_phrases.append(phrase.lower())

        # Count and get most common
        phrase_counts = Counter(filtered_cap_phrases)
        top_cap_phrases = [phrase for phrase, count in phrase_counts.most_common(5) if count >= 1]
        keywords.extend(top_cap_phrases[:2])

        # Strategy 3: Important domain-specific multi-word phrases (2-3 words)
        # Extract all 2-3 word phrases
        content_lower = clean_content.lower()
        two_word = re.findall(r'\b[a-z]+\s+[a-z]+\b', content_lower)
        three_word = re.findall(r'\b[a-z]+\s+[a-z]+\s+[a-z]+\b', content_lower)

        # Common stop words to filter out
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'from', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'our', 'we', 'they'}

        # Filter phrases
        filtered_two_word = []
        for phrase in two_word:
            words = phrase.split()
            # Skip if contains stop words or is too generic
            if not any(w in stop_words for w in words) and len(phrase) > 8:
                filtered_two_word.append(phrase)

        filtered_three_word = []
        for phrase in three_word:
            words = phrase.split()
            # Allow one stop word in three-word phrases (middle word)
            if words[0] not in stop_words and words[-1] not in stop_words and len(phrase) > 12:
                filtered_three_word.append(phrase)

        # Count and get most frequent
        phrase_counts = Counter(filtered_two_word + filtered_three_word)

        # Get top phrases (must appear at least twice)
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(15) if count >= 2]

        # Add diverse phrases (not too similar)
        for phrase in top_phrases:
            if len(keywords) >= 7:
                break
            # Check if this phrase is distinct from existing keywords
            is_distinct = True
            for existing in keywords:
                # Skip if too similar
                if phrase in existing.lower() or existing.lower() in phrase:
                    is_distinct = False
                    break
            if is_distinct:
                keywords.append(phrase)

        # If we don't have enough keywords, add some generic domain terms
        if len(keywords) < 5:
            generic_terms = ['clinical research', 'medical study', 'health outcomes',
                           'data analysis', 'patient care', 'treatment effectiveness', 'research methods']
            needed = 5 - len(keywords)
            keywords.extend(generic_terms[:needed])

        # Clean up keywords: remove newlines, extra whitespace, and very short words
        cleaned_keywords = []
        for keyword in keywords[:10]:  # Process more to have extras after filtering
            # Remove newlines and normalize whitespace
            cleaned = ' '.join(keyword.split())

            # Split if it's actually multiple concepts joined (e.g., "confounding collider")
            # Check if it contains multiple distinct technical terms
            words = cleaned.split()

            # Skip single-letter "words" or very short meaningless words
            if len(cleaned) <= 2:
                continue
            if any(w in cleaned.lower() for w in ['old', 'new', 'the', 'and', 'for']):
                # If it contains these but isn't ONLY these, try to clean it
                if len(words) > 1:
                    # Remove the unwanted words
                    filtered_words = [w for w in words if w.lower() not in ['old', 'new', 'the', 'and', 'for']]
                    if filtered_words:
                        cleaned = ' '.join(filtered_words)
                    else:
                        continue  # Skip entirely if nothing left
                else:
                    continue  # Skip if it's just "old" or "new"

            # Avoid very long phrases (likely multiple concepts joined)
            if len(words) > 5:
                # Take first 3-4 words only
                cleaned = ' '.join(words[:4])

            cleaned_keywords.append(cleaned)

            # Stop once we have enough
            if len(cleaned_keywords) >= 7:
                break

        # Ensure we have 5-7 keywords
        if len(cleaned_keywords) < 5:
            # Add fallback if we filtered too many
            fallback = ['clinical research', 'medical data', 'health outcomes', 'data analysis']
            for term in fallback:
                if len(cleaned_keywords) >= 5:
                    break
                if term not in cleaned_keywords:  # Avoid duplicates
                    cleaned_keywords.append(term)

        return cleaned_keywords[:7]

async def test_single_file(file_path: Path, output_dir: Path):
    """Test extraction on a single file and save output."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print('='*60)

    embedding_service = MockEmbeddingService()
    chat_service = MockChatService()
    processor = DocumentProcessor(embedding_service, chat_service)

    # Process document
    processed_doc = await processor.process_document(file_path)

    if not processed_doc:
        print(f"❌ Failed to process {file_path.name}")
        return

    # Create output filenames
    base_name = file_path.stem
    metadata_file = output_dir / f"{base_name}_metadata.json"
    content_file = output_dir / f"{base_name}_content.txt"
    chunks_file = output_dir / f"{base_name}_chunks.json"
    summary_file = output_dir / f"{base_name}_summary.json"

    # 1. Save metadata
    metadata_dict = {
        "title": processed_doc.metadata.title,
        "author": processed_doc.metadata.author,
        "created_at": processed_doc.metadata.created_at.isoformat() if processed_doc.metadata.created_at else None,
        "file_type": processed_doc.metadata.file_type.value,
        "file_path": processed_doc.metadata.file_path,
        "file_size": processed_doc.metadata.file_size,
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_file}")

    # 2. Save full content
    with open(content_file, 'w') as f:
        f.write(processed_doc.full_content)
    print(f"✅ Content saved to: {content_file}")
    print(f"   Content length: {len(processed_doc.full_content)} characters")

    # 3. Save chunks info
    chunks_info = {
        "whole_file_chunks": {
            "count": len(processed_doc.whole_file_chunks),
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                }
                for c in processed_doc.whole_file_chunks
            ]
        },
        "page_chunks": {
            "count": len(processed_doc.page_chunks),
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                }
                for c in processed_doc.page_chunks
            ]
        },
        "token_chunks": {
            "count": len(processed_doc.token_chunks),
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                }
                for c in processed_doc.token_chunks
            ]
        }
    }
    with open(chunks_file, 'w') as f:
        json.dump(chunks_info, f, indent=2)
    print(f"✅ Chunks info saved to: {chunks_file}")
    print(f"   Whole file chunks: {len(processed_doc.whole_file_chunks)}")
    print(f"   Page chunks: {len(processed_doc.page_chunks)}")
    print(f"   Token chunks: {len(processed_doc.token_chunks)}")

    # 4. Save summary and keywords
    summary_dict = {
        "summary": processed_doc.summary,
        "keywords": processed_doc.keywords
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")
    print(f"   Summary: {processed_doc.summary[:100]}...")
    print(f"   Keywords: {', '.join(processed_doc.keywords)}")

async def main():
    """Test extraction on all supported documents."""
    documents_path = Path(settings.documents_path)
    output_dir = Path("extraction_output")
    output_dir.mkdir(exist_ok=True)

    # Find all supported files
    supported_extensions = [".pdf", ".docx", ".pptx"]
    files = [
        f for f in documents_path.iterdir()
        if f.suffix.lower() in supported_extensions
    ]

    print(f"\nFound {len(files)} documents to test")
    print(f"Output directory: {output_dir.absolute()}\n")

    for file_path in files:
        try:
            await test_single_file(file_path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    print(f"\n{'='*60}")
    print("✅ Testing complete!")
    print(f"Review outputs in: {output_dir.absolute()}")
    print('='*60)

if __name__ == "__main__":
    asyncio.run(main())