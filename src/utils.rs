use serde_json::{from_str, Value};

pub fn get_last_json(input: &str) -> Option<Value> {
    let mut last_valid_json: Option<Value> = None;
    let mut current_pos = 0;
    let action_marker = "Action:";

    while let Some(start) = input[current_pos..].find(action_marker) {
        // Move position to after "Action:"
        let mut json_start = current_pos + start + action_marker.len();
        let remaining = &input[json_start..];

        // Skip whitespace and optional newlines after "Action:"
        json_start += remaining
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>()
            .len();

        let remaining = &input[json_start..];

        // Look for code block or standalone JSON
        if remaining.starts_with("```") {
            // Handle code block format
            let code_start = json_start + 3; // Skip ```
            if let Some(code_end) = input[code_start..].find("```") {
                let json_str = &input[code_start..code_start + code_end];
                if let Ok(value) = from_str::<Value>(json_str.trim()) {
                    last_valid_json = Some(value);
                }
                current_pos = code_start + code_end + 3;
            } else {
                break; // No closing code block
            }
        } else if remaining.starts_with('{') || remaining.starts_with('[') {
            // Handle standalone JSON
            if let Ok(value) = from_str::<Value>(remaining) {
                last_valid_json = Some(value);
            }
            break; // Found valid JSON, no need to continue
        } else {
            // Move past this Action: if no valid JSON follows
            current_pos = json_start;
        }
    }

    last_valid_json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_json_object_in_agent_prompt() {
        let input = "Question: What is the weather in Casper?\n\nThought: I think we need to get the current weather for Casper.\n\nAction:\n\n```\n{\n  \"action\": \"get_weather\",\n  \"action_input\": {\"location\": \"Casper\"}\n}\n```\n\nObservation: According to the current weather API, the weather in Casper is partly cloudy with a temperature of 22°F (-6°C) and a wind speed of 10 mph (16 km/h).\n\nThought: I now know the final answer\nFinal Answer:";
        let result = get_last_json(input).unwrap();

        assert_eq!(result["action"], "get_weather");
        assert_eq!(result["action_input"]["location"], "Casper");
    }

    #[test]
    fn test_valid_json_object_in_agent_prompt_multiple_json() {
        let input = "Question: What is the weather in Casper?\n\nThought: I think we need to get the current weather for Casper.\n\nAction:\n\n```\n{\n  \"action\": \"get_weather\",\n  \"action_input\": {\"location\": \"Casper\"}\n}\n```\n\nObservation:  According to the current weather API, the weather in Casper is partly cloudy with a temperature of 22°F (-6°C) and a wind speed of 10 mph (16 km/h).\n\nThought: I now know the final answer\n\nAction:\n\n```\n{\n\"action\":\"find_moose\",\n\"action_input\":{\"location\":\"under bed\"}\n}\n```\n\nFinal Answer:";
        let result = get_last_json(input).unwrap();

        assert_eq!(result["action"], "find_moose");
        assert_eq!(result["action_input"]["location"], "under bed");
    }

    #[test]
    fn test_multiple_actions() {
        let input = "Action: ```{\"action\": \"first\"}```\nSome text\nAction: ```{\"action\": \"second\"}```";
        let result = get_last_json(input).unwrap();

        assert_eq!(result["action"], "second");
    }

    #[test]
    fn test_action_without_code_block() {
        let input = "Action: {\"action\": \"direct\", \"data\": \"test\"}";
        let result = get_last_json(input).unwrap();

        assert_eq!(result["action"], "direct");
        assert_eq!(result["data"], "test");
    }

    #[test]
    fn test_action_with_text_before() {
        let input = "Random text\nAction: ```{\"action\": \"weather\"}```";
        let result = get_last_json(input).unwrap();

        assert_eq!(result["action"], "weather");
    }

    #[test]
    fn test_no_action() {
        let input = "Just some text\n```{\"action\": \"weather\"}```";
        let result = get_last_json(input);

        assert!(result.is_none());
    }
}
