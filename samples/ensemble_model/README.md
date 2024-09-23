


```
model_repository/
├── gpt2 <--------- (vLLM model)
│   ├── 1
│   │   └── model.json
│   └── config.pbtxt
├── prefix_model (post-processing python model)
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── uppercase_model (pre-processing python model)
    ├── 1
    │   └── model.py
    └── config.pbtxt
```