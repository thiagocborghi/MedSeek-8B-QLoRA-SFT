# MedSeek-8B-QLoRA-SFT

MedSeek-8B-QLoRA-SFT is a medical language model based on DeepSeek-R1-8B, fine-tuned with QLoRA and SFT for clinical reasoning, diagnosis, and medical QA.

## Configuration

- `configs/training.yaml`
- `configs/inference.yaml`
- `configs/prompts.yaml`

## Project Structure

```
MedSeek-8B-QLoRA-SFT/
│── configs/                     
│   ├── training.yaml            
│   ├── inference.yaml           
│   ├── prompts.yaml             
│
│── src/                          
│   ├── train.py                 
│   ├── inference.py             
│   ├── data_loader.py           
│   ├── model.py                 
│   ├── utils.py                  
│
│── logs/                         
│── experiments/                  
```

