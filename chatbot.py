from chatterbot import ChatBot

chatbot = ChatBot("Chatbot Assistant",  # Initialize chatbot
                  storage_adapter='chatterbot.storage.SQLStorageAdapter',
                  database_uri='sqlite:///db.sqlite3',
                  logic_adapters=[
                      {
                          'import_path': 'custom_adapter.NeuralNetAdapter.NeuralNetAdapter'
                      }
                  ])
