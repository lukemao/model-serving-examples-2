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
    
    try:
        config = T5Config.from_pretrained(model_name)
        try:
            print('retrieve saved model')
            tokenizer = T5Tokenizer.from_pretrained('/tmp/'+model_name+'_tokenizer')
            model = TFT5ForConditionalGeneration.from_pretrained('/tmp/'+model_name+'_model', config=config)
        except:
            print('loading and save models')
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = TFT5ForConditionalGeneration.from_pretrained(model_name, config=config)
            tokenizer.save_pretrained('/tmp/'+model_name+'_tokenizer')
            model.save_pretrained('/tmp/'+model_name+'_model')

        task_specific_config = getattr(model.config, "task_specific_params", {})
        translation_config = task_specific_config.get("translation_en_to_de", {})
        model.config.update(translation_config)
        preprocess_text = sentence.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        input_ids = tokenizer.encode(t5_prepared_Text, return_tensors="tf")
        outputs = model.generate(input_ids=input_ids,
                                 max_length=50,)
        response = '{ output:'+tokenizer.decode(outputs[0])+'\n}'
        print(response)
    except Exception as e:
        response = 'Something went wrong'
        print(e)
    return response