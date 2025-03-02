pub fn get_weather(location: &str) -> String {
    println!("IN GET WEATHER");
    format!("\n\nAPI ANSWER:\n\n```\nThe weather in {} is 23 F with 30 mph winds coming from the NorthWest and partly cloudy conditions.\n```\n\n", location)
}
