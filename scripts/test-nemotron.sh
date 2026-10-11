#!/usr/bin/env bash
set -euo pipefail

# Load NVIDIA_API_KEY from deploy/.env
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/.env"
set -a
source "$ENV_FILE"
set +a
: "${NVIDIA_API_KEY:?NVIDIA_API_KEY is not set in $ENV_FILE}"

invoke_url='https://integrate.api.nvidia.com/v1/chat/completions'

payload=$(cat <<'JSON'
{
  "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
  "messages": [{"content":"hello","role":"user"}],
  "temperature": 1,
  "top_p": 0.95,
  "max_tokens": 16384,
  "reasoning_budget": 16384,
  "chat_template_kwargs": {"enable_thinking":true},
  "stream": false,
  "tools": [{"type":"function","function":{"name":"describe_harry_potter_character","description":"Returns information and images of Harry Potter characters.","parameters":{"type":"object","properties":{"name":{"type":"string","enum":["Harry James Potter","Hermione Jean Granger","Ron Weasley","Fred Weasley","George Weasley","Bill Weasley","Percy Weasley","Charlie Weasley","Ginny Weasley","Molly Weasley","Arthur Weasley","Neville Longbottom","Luna Lovegood","Draco Malfoy","Albus Percival Wulfric Brian Dumbledore","Minerva McGonagall","Remus Lupin","Rubeus Hagrid","Sirius Black","Severus Snape","Bellatrix Lestrange","Lord Voldemort","Cedric Diggory","Nymphadora Tonks","James Potter"],"description":"Name of the Harry Potter character"}},"required":["name"]}}},{"type":"function","function":{"name":"name_a_color","description":"A tool that returns a bunch of color names for a given color_hex.","parameters":{"type":"object","properties":{"color_hex":{"type":"string","description":"A hexadecimal color value which must be represented as a string."}},"required":["color_hex"]}}}],
  "tool_choice": "auto"
}
JSON
)

curl -sS -N \
  --request POST \
  --url "$invoke_url" \
  --header "Authorization: Bearer ${NVIDIA_API_KEY}" \
  --header "Accept: application/json" \
  --header "Content-Type: application/json" \
  --data "$payload"
