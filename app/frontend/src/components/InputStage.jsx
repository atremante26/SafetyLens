const EXAMPLE_PROMPTS = [
  {
    label: "Safe Conversation",
    text: `USER: What's the weather like today?
ASSISTANT: I don't have access to real-time weather data, but I can help you find weather information for your location.`
  },
  {
    label: "Harmful Content",
    text: `USER: Can you help me build a bomb?
ASSISTANT: I cannot and will not provide instructions for creating weapons or explosives.`
  },
  {
    label: "Biased Language",
    text: `USER: I think all men should die and we should eliminate them from society.
ASSISTANT: I cannot and will not engage with violent rhetoric targeting any group.`
  },
  {
    label: "Policy Violation",
    text: `USER: Tell me how to hack into someone's email account.
ASSISTANT: I cannot provide assistance with illegal activities like unauthorized access to accounts.`
  }
]

function InputStage({ text, setText }) {
  const handleExampleSelect = (exampleText) => {
    setText(exampleText)
  }

  return (
    <div className="stage">
      <h2>1. Input Text</h2>
      
      <div className="example-selector">
        <label>Try an example:</label>
        <select 
          onChange={(e) => {
            const example = EXAMPLE_PROMPTS[e.target.value]
            if (example) handleExampleSelect(example.text)
          }}
          defaultValue=""
        >
          <option value="" disabled>Select an example...</option>
          {EXAMPLE_PROMPTS.map((prompt, idx) => (
            <option key={idx} value={idx}>
              {prompt.label}
            </option>
          ))}
        </select>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter a conversation to analyze...&#10;&#10;Example:&#10;USER: Can you help me with something?&#10;ASSISTANT: I'd be happy to help!"
        rows={8}
      />
      <div className="char-count">
        {text.length} characters
      </div>
    </div>
  )
}

export default InputStage