use std::collections::HashSet;
use std::path::Path;
use tree_sitter::{Language, Parser as TSParser, Query, QueryCursor};
use streaming_iterator::StreamingIterator;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub range: TextRange,
    pub content: String,
    pub references: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Class,
    Interface,
    Trait,
    Module,
    Unknown(String),
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy)]
pub struct TextRange {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

pub trait LanguageParser {
    fn language(&self) -> Language;
    fn query_source(&self) -> &'static str;
    
    fn extract_symbols(&self, content: &str) -> anyhow::Result<Vec<Symbol>> {
        let mut parser = TSParser::new();
        parser.set_language(&self.language())?;
        let tree = parser.parse(content, None).ok_or_else(|| anyhow::anyhow!("Failed to parse"))?;
        let root = tree.root_node();
        
        let query = Query::new(&self.language(), self.query_source())?;
        let mut cursor = QueryCursor::new();
        let mut captures = cursor.captures(&query, root, content.as_bytes());
        
        let mut symbols = Vec::new();
        while let Some(item) = captures.next() {
            let (m, capture_index) = item;
            let capture = m.captures[*capture_index];
            let capture_name = query.capture_names()[capture.index as usize];
            
            if capture_name == "symbol" {
                let node = capture.node;
                let mut name = String::new();
                let kind = self.parse_kind(node.kind());
                
                for c in m.captures {
                    if query.capture_names()[c.index as usize] == "name" {
                        name = content[c.node.start_byte()..c.node.end_byte()].to_string();
                    }
                }
                
                if name.is_empty() {
                    name = node.child_by_field_name("name")
                        .map(|n: tree_sitter::Node| content[n.start_byte()..n.end_byte()].to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                }
                
                let start = node.start_position();
                let end = node.end_position();
                
                symbols.push(Symbol {
                    name,
                    kind,
                    range: TextRange {
                        start_line: start.row as u32 + 1,
                        start_col: start.column as u32,
                        end_line: end.row as u32 + 1,
                        end_col: end.column as u32,
                    },
                    content: content[node.start_byte()..node.end_byte()].to_string(),
                    references: self.find_references(node, content),
                });
            }
        }
        
        Ok(symbols)
    }

    fn parse_kind(&self, kind: &str) -> SymbolKind;
    
    fn find_references(&self, node: tree_sitter::Node, content: &str) -> Vec<String> {
        let mut refs = Vec::new();
        let mut cursor = node.walk();
        let mut reached_root = false;
        while !reached_root {
            let sub_node = cursor.node();
            if sub_node.kind() == "call_expression" || sub_node.kind() == "call" {
                if let Some(fn_node) = sub_node.child_by_field_name("function") {
                    refs.push(content[fn_node.start_byte()..fn_node.end_byte()].to_string());
                }
            }

            if cursor.goto_first_child() {
                continue;
            }
            while !cursor.goto_next_sibling() {
                if !cursor.goto_parent() || cursor.node() == node {
                    reached_root = true;
                    break;
                }
            }
        }
        refs
    }
}

pub struct RustParser;
impl LanguageParser for RustParser {
    fn language(&self) -> Language { tree_sitter_rust::LANGUAGE.into() }
    fn query_source(&self) -> &'static str {
        r#"
        (function_item name: (identifier) @name) @symbol
        (struct_item name: (type_identifier) @name) @symbol
        (impl_item) @symbol
        (trait_item name: (type_identifier) @name) @symbol
        "#
    }
    fn parse_kind(&self, kind: &str) -> SymbolKind {
        match kind {
            "function_item" => SymbolKind::Function,
            "struct_item" => SymbolKind::Struct,
            "impl_item" => SymbolKind::Method,
            "trait_item" => SymbolKind::Trait,
            _ => SymbolKind::Unknown(kind.to_string()),
        }
    }
}

pub struct PythonParser;
impl LanguageParser for PythonParser {
    fn language(&self) -> Language { tree_sitter_python::LANGUAGE.into() }
    fn query_source(&self) -> &'static str {
        r#"
        (function_definition name: (identifier) @name) @symbol
        (class_definition name: (identifier) @name) @symbol
        "#
    }
    fn parse_kind(&self, kind: &str) -> SymbolKind {
        match kind {
            "function_definition" => SymbolKind::Function,
            "class_definition" => SymbolKind::Class,
            _ => SymbolKind::Unknown(kind.to_string()),
        }
    }
}

pub struct JavaScriptParser;
impl LanguageParser for JavaScriptParser {
    fn language(&self) -> Language { tree_sitter_javascript::LANGUAGE.into() }
    fn query_source(&self) -> &'static str {
        r#"
        (function_declaration name: (identifier) @name) @symbol
        (class_declaration name: (identifier) @name) @symbol
        (method_definition name: (property_identifier) @name) @symbol
        "#
    }
    fn parse_kind(&self, kind: &str) -> SymbolKind {
        match kind {
            "function_declaration" => SymbolKind::Function,
            "class_declaration" => SymbolKind::Class,
            "method_definition" => SymbolKind::Method,
            _ => SymbolKind::Unknown(kind.to_string()),
        }
    }
}

pub struct Chunk {
    pub text: String,
    pub line: u32,
    pub name: String,
    pub references: Vec<String>,
}

pub fn structural_chunk(content: &str, file_path: &Path) -> Vec<Chunk> {
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let parser: Box<dyn LanguageParser> = match extension {
        "rs" => Box::new(RustParser),
        "py" => Box::new(PythonParser),
        "js" | "jsx" | "ts" | "tsx" => Box::new(JavaScriptParser),
        _ => {
            return chunk_text_line_aware(content, 500, 50)
                .into_iter()
                .map(|(text, line)| Chunk {
                    text,
                    line,
                    name: String::new(),
                    references: Vec::new(),
                })
                .collect();
        }
    };

    match parser.extract_symbols(content) {
        Ok(symbols) => symbols.into_iter().map(|s| Chunk {
            text: format!("[Context: {:?} {}]\n{}", s.kind, s.name, s.content),
            line: s.range.start_line,
            name: s.name,
            references: s.references,
        }).collect(),
        Err(_) => chunk_text_line_aware(content, 500, 50)
            .into_iter()
            .map(|(text, line)| Chunk {
                text,
                line,
                name: String::new(),
                references: Vec::new(),
            })
            .collect(),
    }
}

pub fn chunk_text_line_aware(text: &str, chunk_size: usize, overlap: usize) -> Vec<(String, u32)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current_line = 0;

    while current_line < lines.len() {
        let mut current_chunk = String::new();
        let mut lines_in_chunk = 0;
        let start_line = current_line as u32 + 1;

        while current_line < lines.len() && current_chunk.len() < chunk_size {
            current_chunk.push_str(lines[current_line]);
            current_chunk.push('\n');
            current_line += 1;
            lines_in_chunk += 1;
        }

        if !current_chunk.trim().is_empty() {
            chunks.push((current_chunk, start_line));
        }

        if current_line < lines.len() {
            let back = lines_in_chunk.min(overlap / 50 + 1);
            current_line = current_line.saturating_sub(back).max(current_line - lines_in_chunk + 1);
        }
    }

    chunks
}

pub fn is_supported_file(path: &Path) -> bool {
    let extensions: HashSet<&'static str> = [
        "py", "rs", "js", "ts", "jsx", "tsx", "md", "txt", "nix", "go", "c", "cpp", "h", "hpp",
    ]
    .into_iter()
    .collect();
    
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    extensions.contains(ext)
}
