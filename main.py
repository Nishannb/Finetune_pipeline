import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from datasets import Dataset
from groq import Groq
from typing import List, Dict
import os
import sample
from dotenv import load_dotenv

load_dotenv()

groq_api = os.environ["GROQ_API_KEY"]

class DataCollector:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        
    def get_llm_response(self, prompt: str) -> str:
        """Get response from Groq LLM"""
        try:
            system_prompt = """You are a robotic trip planner. Whenever, you are asked a question, 
            you go search the information on internet and reply based on response you get from internet. 
            If you cannot go on internet or find any information on internet, say I cannot do this.
            Also, Always include the latitude and longitude in square brackets for any location mentioned in your response.
            Your output format should be not more than 250 words for each."""
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",  # 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            # print(f"response:  {response.choices[0].message.content}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting Groq response: {e}")
            return ""

    def collect_data(self, prompts: List[str]) -> List[Dict]:
        """Collect prompt-response pairs"""
        dataset = []
        for prompt in prompts:
            response = self.get_llm_response(prompt)
            if response:
                dataset.append({
                    "prompt": prompt,
                    "response": response
                })
        return dataset


class SmallModelTrainer:
    def __init__(self, model_name: str = "google/flan-t5-small", checkpoint_dir: str = None ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
            self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
        else:
            print("Starting training from scratch.")
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.generation_config.use_cache = False
        
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset with proper tokenization"""
        def preprocess_function(examples):
            inputs = [f"Answer the question: {prompt}" for prompt in examples['input_text']]
            targets = examples['target_text']
            
            model_inputs = self.tokenizer(
                inputs, 
                max_length=512, 
                truncation=True, 
                padding=True,  # Add padding
                return_tensors='pt'  # Return PyTorch tensors
            )
            
            labels = self.tokenizer(
                targets, 
                max_length=512, 
                truncation=True, 
                padding=True,
                return_tensors='pt'
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = Dataset.from_dict({
            "input_text": [item["prompt"] for item in data],
            "target_text": [item["response"] for item in data]
        })
        
        return dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    def train(self, dataset: Dataset, output_dir: str = "trained_model"):
        """Train the model from scratch and save checkpoints in .bin format."""
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=25,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            save_safetensors=False,  # Save checkpoints as .bin files
            overwrite_output_dir=True,  # Force overwrite to start fresh
            use_cpu=True,
            resume_from_checkpoint=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        trainer.train(resume_from_checkpoint=True )  # Start training from scratch
        trainer.save_model()  # Save the final model
        print("Training complete. Model and checkpoints saved in PyTorch .bin format.")

    def inference(self, prompt: str) -> str:
        """Generate response for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Example usage
    api_key = groq_api
    
    # 1. Collect data
    collector = DataCollector(api_key)

    dataset = collector.collect_data(sample.sample_prompts6)
    
    # 2. Train small model
    trainer = SmallModelTrainer(checkpoint_dir="trained_model/checkpoint-325")  # Resumes from checkpoint

    hf_dataset = trainer.prepare_dataset(dataset)
    trainer.train(hf_dataset)
    
     # 3. Test inference with similar queries
    test_prompts = [
        "Recommend a tourist spot close to Yurakucho Station",
        "How far is Roppongi Hills from Yokohama"
    ]
    
    for prompt in test_prompts:
        response = trainer.inference(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()