def play_file_qa(fname, outfile, tokenizer, model, trial_num=5,
              steps=256, temperature=0.6, do_sample=True,
              top_k=5, top_p=None, until_n_eos=1, limit_samples=-1):
    print(f"Generating by prompts from {fname}...")
    # full_stop_input_ids = get_input_ids_of_stop_tokens(tokenizer)
    full_stop_input_ids = None
    count = 0
    with hopen(fname) as f:
        with open(outfile, "w", encoding="utf-8") as fout:
            for line in f:
                count += 1
                if limit_samples > 0 and count >= limit_samples:
                    print(f"Reach limit_samples: {limit_samples}, stop.")
                    break
                try:
                    xAndY = ''

                    jl = json.loads(line)
                    text = jl['page_info']['query'].strip()
                    # print('prompt/query: ', text)
                    xAndY=xAndY+'prompt/query: '+text+'\n'
                    xAndY=xAndY+f'tokens: {tokenizer.tokenize(text)}'+'\n'
                    # print('origin content', jl['page_info']['query'][:steps])
                    # print("ground truth answer: {}".format(jl['page_info']['answer']))
                    xAndY=xAndY+"ground truth answer: {}".format(jl['page_info']['answer'])+'\n'
                    input_ids = torch.tensor(tokenizer(text)['input_ids']).unsqueeze(0).long()
                    # print("input_ids: {}".format(input_ids))
                    
                    for i in range(trial_num):
                        y = sample_generate(model,
                                            input_ids=input_ids.to(model.device),
                                            steps=steps, temperature=temperature, do_sample=do_sample,
                                            top_k=top_k,
                                            top_p=top_p, 
                                            eos=tokenizer.eos_token_id,
                                            until_n_eos=until_n_eos)

                        completion = ''.join(tokenizer.decode(y))
                        completion.replace('##', '')

                        # print(f'[{i}]: {completion}')
                        xAndY=xAndY+f'[{i}]: {completion}'+'\n'

                        json_str = json.dumps(
                            {
                                'prompt/query: ': f'{text}', 
                                'ground truth answer: ': 'ground truth answer: {}'.format(jl['page_info']['answer']),
                                'answer: ': f'[{i}]: {completion}'
                            }, 
                            ensure_ascii=False
                        )

                        fout.write(json_str + "\n")
                    print(xAndY)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())