def zeroshot_t5(request):
    from transformers import (
        TFT5ForConditionalGeneration,
        T5Tokenizer,
        T5Config,
    )
    request_json = request.get_json()
    print(request_json)
    
    sentence = request_json['sentence']
    model_name = request_json['model_name']

    config = T5Config.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = TFT5ForConditionalGeneration.from_pretrained(model_name, config=config)
        
    task_specific_config = getattr(model.config, "task_specific_params", {})
    translation_config = task_specific_config.get("translation_en_to_de", {})
    model.config.update(translation_config)
        
    t5_prepared_Text = sentence.strip().replace("\n","")
        
    input_ids = tokenizer.encode(t5_prepared_Text, return_tensors="tf")
    outputs = model.generate(input_ids=input_ids,
                                 max_length=50,)
        
    response = '{ output:'+tokenizer.decode(outputs[0])+'\n}'
    return response