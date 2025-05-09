import os
from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
import torch
import numpy as np

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

class NeuralNetAdapter(LogicAdapter):
    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)

        # Load model and data
        FILE = "data/data.pth"
        data = torch.load(FILE)

        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

        # Load intents from text files
        self.load_intents_from_files("train_data")

    def load_intents_from_files(self, directory):
        self.intents = []

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                tag = filename.replace(".txt", "")
                intent = self.parse_intent_file(os.path.join(directory, filename), tag)
                if intent:
                    self.intents.append(intent)

    def parse_intent_file(self, file_path, tag):
        patterns = []
        responses = []

        with open(file_path, "r", encoding="windows-1252") as file:
            lines = file.readlines()

        current_section = None
        current_response = []

        for line in lines:
            line = line.strip()

            if line.startswith("intent:"):
                continue
            elif line.startswith("patterns:"):
                current_section = "patterns"
            elif line.startswith("responses:"):
                current_section = "responses"
            elif current_section == "patterns" and line.startswith("- "):
                patterns.append(line[2:].strip())
            elif current_section == "responses":
                if line.startswith("- "):  # Start of a new response
                    if current_response:
                        responses.append('\n'.join(current_response))
                    current_response = [line[2:].strip()]
                elif line.startswith("-- "):  # Sub-line of the response. More like a line break
                    current_response.append(line[3:].strip())
                elif line == "":
                    continue

        if current_response:
            responses.append('\n'.join(current_response))

        if responses:
            return {
                "tag": tag,
                "patterns": patterns,
                "responses": responses
            }
        return None


    def can_process(self, statement):
        return True

    def process(self, statement, additional_response_selection_parameters=None):
        sentence = statement.text
        tokens = tokenize(sentence)
        bow = bag_of_words(tokens, self.all_words)
        bow_tensor = torch.from_numpy(bow).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(bow_tensor)
            _, predicted = torch.max(output, dim=1)
            tag = self.tags[predicted.item()]
            probs = torch.softmax(output, dim=1)
            confidence = probs[0][predicted.item()]

        if confidence.item() > 0.75:
            for intent in self.intents:
                if intent["tag"] == tag:
                    response = np.random.choice(intent["responses"])
                    break
        else:
            response = "I'm not sure I understand. Can you try again?"

        return Statement(text=response)