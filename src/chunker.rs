use std::collections::HashSet;
use std::path::Path;
use tree_sitter::Parser as TSParser;

pub struct Chunk {
    pub text: String,
    pub line: u32,
    pub name: String,
    pub references: Vec<String>,
}

pub fn structural_chunk(content: &str, file_path: &Path) -> Vec<Chunk> {
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let language = match extension {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        _ => None,
    };

    let Some(lang) = language else {
        return chunk_text_line_aware(content, 500, 50)
            .into_iter()
            .map(|(text, line)| Chunk {
                text,
                line,
                name: String::new(),
                references: Vec::new(),
            })
            .collect();
    };

    let mut parser = TSParser::new();
    parser.set_language(&lang).expect("Error loading grammar");
    let tree = parser.parse(content, None).expect("Error parsing");

    let mut chunks = Vec::new();
    let mut cursor = tree.walk();
    let root = tree.root_node();

    for node in root.children(&mut cursor) {
        let kind = node.kind();
        let mut symbol_name = String::new();
        let mut refs = Vec::new();

        if kind == "struct_item"
            || kind == "impl_item"
            || kind == "class_definition"
            || kind == "function_definition"
            || kind == "function_item"
        {
            symbol_name = node
                .child_by_field_name("name")
                .or_else(|| node.child_by_field_name("declarator"))
                .map(|n| &content[n.start_byte()..n.end_byte()])
                .unwrap_or("unknown")
                .to_string();

            // Extract references (basic call detection)
            let mut sub_cursor = node.walk();
            let mut reached_root = false;
            while !reached_root {
                let sub_node = sub_cursor.node();
                if sub_node.kind() == "call_expression" || sub_node.kind() == "call" {
                    if let Some(fn_node) = sub_node.child_by_field_name("function") {
                        refs.push(content[fn_node.start_byte()..fn_node.end_byte()].to_string());
                    }
                }

                if sub_cursor.goto_first_child() {
                    continue;
                }
                while !sub_cursor.goto_next_sibling() {
                    if !sub_cursor.goto_parent() || sub_cursor.node() == node {
                        reached_root = true;
                        break;
                    }
                }
                if reached_root {
                    break;
                }
            }
        }

        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let text = &content[start_byte..end_byte];
        let start_line = node.start_position().row as u32 + 1;

        if text.len() > 1000 {
            for (sub_text, sub_line) in chunk_text_line_aware(text, 1000, 100) {
                let context_text = if !symbol_name.is_empty() {
                    format!("[Context: {} {}]\n{}", kind, symbol_name, sub_text)
                } else {
                    sub_text
                };
                chunks.push(Chunk {
                    text: context_text,
                    line: start_line + sub_line - 1,
                    name: symbol_name.clone(),
                    references: refs.clone(),
                });
            }
        } else if text.len() > 20 {
            let context_text = if !symbol_name.is_empty() {
                format!("[Context: {} {}]\n{}", kind, symbol_name, text)
            } else {
                text.to_string()
            };
            chunks.push(Chunk {
                text: context_text,
                line: start_line,
                name: symbol_name,
                references: refs,
            });
        }
    }

    if chunks.is_empty() && !content.trim().is_empty() {
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

    chunks
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
