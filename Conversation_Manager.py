from openai import OpenAI
import tiktoken
import json
from datetime import datetime
import os


class Conversation_Manager:

    """
    Initializes the ConversationManager with the specified configurations
    Initializes client, sets base URL, manages history file, and sets defaults for conversation parameters
    
    Parameters:
        api_key (str): API key for authentication with the OpenAI service
        base_url (str): Base URL for the OpenAI API
        history_file (str, optional): Path to the file where the conversation history will be stored
                                    If None, a new file is created with a timestamp
        default_model (str): Default model to use for generating responses
        default_temperature (float): Default randomness in the response generation
        default_max_tokens (int): Default maximum number of tokens per response
        token_budget (int): Total token budget for the conversation

    Returns:
        None
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1", history_folder="Past_Conversations", history_file=None, default_model="gpt-3.5-turbo", default_temperature=0.7, default_max_tokens=150, token_budget=4096):
        # Use OpenAI as default key
        self.client = OpenAI(api_key=api_key)
        self.base_url = base_url

        # Check if history folder exists
        if not os.path.exists(history_folder):
            os.makedirs(history_folder)
        
        # Check if chat history file exists and set directory to history_folder
        if history_file is None:
            timestamp = datetime.now().strftime("%m%d%y_%H%M%S")
            self.history_file = os.path.join(history_folder, f"conversation_history_{timestamp}.json")
        else:
            self.history_file = os.path.join(history_folder, history_file)

        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.token_budget = token_budget

        # Provide default assistants
        self.system_messages = {
            "normal_assistant": "You are a normal, cooperative assistant that does what they are told.",
            "sassy_assistant": "You are a sassy assistant that is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant, always ready to dig deeper. You ask clarifying questions to ensure understanding and approach problems with a step-by-step methodology.",
            "custom": "Enter your custom system message here."
        }

        # Set default persona to be the normal assistant
        self.system_message = self.system_messages["normal_assistant"]

        self.load_conversation_history()


    """
    Counts the number of tokens in a given text

    Parameters:
        text (str): Text to be tokenized and counted
    
    Returns:
        int: Num of tokens in a given text
    """
    def count_tokens(self, text):
        try:
            # Count num of tokens in a given text
            encoding = tiktoken.encoding_for_model(self.default_model)

        except KeyError:
            # Check if model if valid
            print(f"Warning: Model '{self.default_model}' not found. Using 'gpt-3.5-turbo' encoding as default.")
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Calculate num tokens used for a text
        tokens = encoding.encode(text)
        return len(tokens)


    """
    Calculate the total num of tokens used in the conversation history
    
    Returns:
        int: Total num of tokens used across all messages in conversation history
        None: If an error occurs
    """
    def total_tokens_used(self):
        try:
            # Calculate the num of tokens used in the conversation history
            sum = 0
            for message in self.conversation_history:
                sum += self.count_tokens(message["content"])
            return sum
        
        except Exception as e:
            print(f"An unexpected error occurred while calculating the total tokens used: {e}")
            return None
    

    """
    Enforces token budget by removing older messages from conversation history if the total token count exceeds the present budget
    Stops removing messages once the budget is no longer exceeded

    Returns:
        None
    """
    def enforce_token_budget(self):
        try:
            # Check and see if total token count has exceeded present budget
            while self.total_tokens_used() > self.token_budget:

                # If conversation history is only one entry
                if len(self.conversation_history) <= 1:
                    break
                self.conversation_history.pop(1)
            
        except Exception as e:
            print(f"An unexpected error occurred while enforcing the token budget: {e}")


    """
    Set the assistant's persona based on predefined system messages

    Parameters:
        persona (str): Key of the desired persona to be set as current system message
    
    Returns:
        None
    """
    def set_persona(self, persona):
        # Check to see if persona is already predefined
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()

        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")


    """
    Update current persona to 'custom' and set a custom message for the custom persona

    Parameters:
        custom_message (str): Custom message text to set

    Returns:
        None
    """        
    def set_custom_system_message(self, custom_message):
        # Check to see if message is not empty
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        
        # Set persona to custom and update its message
        self.system_messages["custom"] = custom_message
        self.set_persona("custom")


    """
    Updates the system message in the conversation history to reflect the current persona
    
    Returns:
        None
    """
    def update_system_message_in_history(self):
        try:
            # Check to see if conversation history is not empty and if first entry is a system message
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                self.conversation_history[0]["content"] = self.system_message
            
            # Insert a new system message at the beginning of the conversation history
            else:
                self.conversation_history.insert(0, {"role": "system", "content": self.system_message})

        except Exception as e:
            print(f"An unexpected error occurred while updating this system message in the conversation history: {e}")
            

    """
    Generates a chat response using the OpenAI API
    Appends user's prompt and AI's response to conversation history and saves history

    Parameters:
        prompt (str): User input prompt to generate a response for
        temperature (float, optional): Degree of randomness in the response generation
        max_tokens (int, optional): Max num of tokens allowed in the response
        model (str, optional): Model to use for the response generation
    
    Returns:
        str: AI generated response from user's prompt
    """        
    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        # Check if parameters are defined, else set them to their default
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        model = model if model is not None else self.default_model

        # Append prompt to conversation history and enforce the token budget
        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        try:
            # Make a call to the OpenAI API & obtain the response
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            return None
        
        # Retrieve content of first response and append it to conversation history
        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        self.save_conversation_history()

        return ai_response


    """
    Loads the conversation history from a JSON file
    Initializes new default conversation history if encounters error with file locating or decoding
    """
    def load_conversation_history(self):
        try:
            # Load conversation history file to conversation_history
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)

        # Initializes new conversation_history if no file found
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        
        # Initializes new conversation_history if errors while reading old conversation history file
        except json.JSONDecoder:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]
    
    
    """
    Saves the current conversation history to a JSON file
    
    Returns:
        None
    """
    def save_conversation_history(self):
        try:
            # Write conversation_history contents to a file
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        
        except IOError as e:
            print(f"An I/O error occurred while saving the conversation history: {e}")
        
        except Exception as e:
            print(f"An unexpected error occurred while saving the conversation history: {e}")
    

    """
    Resets the conversation history to default state with only the current system message
    Saves reset history to new file

    Returns:
        None
    """
    def reset_conversation_history(self):
        # Reset the conversation history to default state
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        
        try:
            # Attempt to save the reset history to the file
            self.save_conversation_history()
        
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history: {e}")
            
