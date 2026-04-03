-- Database schema for word categorizations (Supabase/PostgreSQL)

-- Main categorizations table (one row per word)
CREATE TABLE word_categorizations (
  id SERIAL PRIMARY KEY,
  public_id UUID REFERENCES kor_word(public_id) UNIQUE NOT NULL,
  lemma TEXT NOT NULL,
  definition TEXT,
  categories JSONB NOT NULL, -- {"하위개념": ["동물", "생물"], "기능": ["애완동물"], "사용맥락": ["일상대화"]}
  processed_at TIMESTAMP NOT NULL,
  model_version TEXT NOT NULL DEFAULT 'exaone3.5:7.8b',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Category lookup index (normalized for fast queries)
CREATE TABLE category_index (
  id SERIAL PRIMARY KEY,
  classification_type TEXT NOT NULL, -- '하위개념', '기능', '사용맥락'
  category_name TEXT NOT NULL,
  public_id UUID REFERENCES kor_word(public_id) NOT NULL,
  lemma TEXT NOT NULL, -- Denormalized for convenience
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_category_lookup ON category_index(classification_type, category_name);
CREATE INDEX idx_public_id ON word_categorizations(public_id);
CREATE INDEX idx_categories_gin ON word_categorizations USING GIN(categories);

-- Query examples:

-- 1. Get all categories for a specific word
-- SELECT categories FROM word_categorizations WHERE public_id = '...';

-- 2. Get all words in a category (normalized index)
-- SELECT public_id, lemma FROM category_index 
-- WHERE classification_type = '하위개념' AND category_name = '동물';

-- 3. Get all categories of a type (using JSONB index)
-- SELECT DISTINCT jsonb_array_elements_text(categories->'하위개녑') 
-- FROM word_categorizations;

-- 4. Get words with multiple categories (JSONB query)
-- SELECT public_id, lemma FROM word_categorizations 
-- WHERE categories @> '{"하위개념": ["동물"]}';

-- 5. Find similar categories (for post-processing)
-- SELECT c1.category_name, c2.category_name, COUNT(*) as overlap
-- FROM category_index c1
-- JOIN category_index c2 ON c1.public_id = c2.public_id
-- WHERE c1.classification_type = c2.classification_type
--   AND c1.category_name < c2.category_name
-- GROUP BY c1.category_name, c2.category_name
-- HAVING COUNT(*) > 5
-- ORDER BY overlap DESC;

-- Import script (Python):
-- 
-- import json
-- from supabase import Client
-- 
-- def import_categorizations(supabase: Client, json_file: str):
--     with open(json_file, 'r', encoding='utf-8') as f:
--         data = json.load(f)
--     
--     # Import to word_categorizations
--     for record in data['categorizations']:
--         supabase.table('word_categorizations').insert({
--             'public_id': record['public_id'],
--             'lemma': record['lemma'],
--             'definition': record['definition'],
--             'categories': record['categories'],
--             'processed_at': record['processed_at'],
--             'model_version': record['model_version']
--         }).execute()
--     
--     # Populate category_index
--     for class_type, categories in data['category_index'].items():
--         for category_name, words in categories.items():
--             for word in words:
--                 supabase.table('category_index').insert({
--                     'classification_type': class_type,
--                     'category_name': category_name,
--                     'public_id': word['public_id'],
--                     'lemma': word['lemma']
--                 }).execute()