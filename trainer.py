import os
from chatbot import chatbot
from chatterbot.trainers import ListTrainer

trainer = ListTrainer(chatbot)

# Logs all user inputs for future feeding
# Manual review and update to train_data files
def log_interaction(user_input, bot_response):
    os.makedirs("logs", exist_ok=True)
    with open("logs/new_data.txt", "a", encoding="utf-8") as f:
        f.write(f"{user_input}|||{bot_response}\n")


def brain(user_input):
    response = chatbot.get_response(user_input)
    if str(response) == "I'm not sure I understand. Can you try again?":
        log_interaction(user_input, str(response))

    return response