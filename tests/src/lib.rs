pub fn add(left: usize, right: usize) -> usize {
    left + right
}
//PRINT TO STDOUT
//cargo test -- --nocapture
//or
//cargo watch "test -- --nocapture"
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_searialization() {
        // Your raw JSON string
        let raw_json = r#"
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a highly advanced assistant. You receive a prompt from a user and relevant excerpts extracted from a text document. You then answer truthfully to the best of your ability. If you do not know the answer, your response is I don't know."
            },
            {
                "role": "user",
                "content": "your_user_prompt"
            },
            {
                "role": "system",
                "content": "Based on the retrieved information from the document, here are the relevant excerpts:\n{{payloads}}\nPlease provide a comprehensive answer to the user's question, integrating insights from these excerpts and your general knowledge."
            }
        ],
        "model": "mixtral-8x7b-32768"
    }
    "#;

        // Deserialize the raw JSON string
        let mut deserialized: Value = serde_json::from_str(raw_json).unwrap();
        println!("{}", serde_json::to_string_pretty(&deserialized).unwrap());

        let payloads: [&str; 3] = ["one", "two", "three"];

        // Update the content field in the second message
        if let serde_json::Value::Array(ref mut messages) = deserialized["messages"] {
            if let serde_json::Value::String(ref mut content) = messages[1]["content"] {
                *content = format!("your_updated_user_prompt");
            }
            if let serde_json::Value::String(ref mut content) = messages[2]["content"] {
                let joined_payloads: String = payloads.join("\n");
                let start = content.find("{{p").unwrap();
                let end = &start + 12;

                content.replace_range(start..end, &joined_payloads)
            }
        }

        // Serialize back to a JSON string or use the deserialized Value directly
        // let updated_json = serde_json::to_string(&deserialized).unwrap();

        //Check after update
        println!("{}", serde_json::to_string_pretty(&deserialized).unwrap());

        // assert_eq!(deserialized, expect);
    }
}
