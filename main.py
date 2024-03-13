from genco.get import *
import hydra

# function for evaluation
def check_label(row):
    if row['image_id'] == None:
        return 0
    elif row['input_text'] in str(row['gt']):
        return 1
    elif (row['input_text'].split(' ')[0] in str(row['gt'])) and (row['input_text'].split(' ')[1] in str(row['gt'])):
        return 1
    else: return 0


# function for main experiment
def experiment(model_id, dataset_name, pos):
    result = []

    for input_text in genco.input_ls:
        first_turn_start = time.time()
        index_list, hypo, distances = genco.first_turn(input_text)
        result_list = genco.get_captions(index_list)
        first_turn_end = time.time()

        multi_turn_start = time.time()
        num = 0
        while genco.multi_turn(input_text, result_list) == "retry":
            index_list, re_hypo, distances, num = genco.re_first_turn(input_text, hypo, num)
            hypo = hypo + ',' + re_hypo
            result_list = genco.get_captions(index_list)
        multi_turn_end = time.time()

        first_turn_time = first_turn_end - first_turn_start
        multi_turn_time = multi_turn_end - multi_turn_start
        result_list = [(input_text, hypo, distances, item[0], item[1], num, first_turn_time, multi_turn_time) for item in result_list]
        result.append(result_list)

    result_df = pd.DataFrame(
        columns=['input_text', 'hypo', 'distances', 'image_id', 'captions', 'iter_num', 'first_turn_time', 'multi_turn_time']
        )
    
    for sublist in result:
        for item in sublist:
            sub_df = pd.DataFrame({
                'input_text' : item[0], 
                'hypo' : item[1], 
                'distances' : item[2],
                'image_id': [item[3]], 
                'captions': [item[4]], 
                'iter_num' : item[5],
                'first_turn_time' : item[6],
                'multi_turn_time' : item[7],
                })
            result_df = pd.concat([result_df, sub_df], ignore_index=True)

    full_df = pd.merge(result_df, genco.gt_df, on='image_id', how='inner')
    full_df['answer'] = full_df.apply(check_label, axis=1)
    
    return full_df




# function for reputational main experiment
@hydra.main(config_path='configs', config_name='config' version_base=None)
def rep_exp(config):
    model_id = config.params.base.model_id
    dataset_name = config.params.base.dataset_name
    pos = config.params.base.pos
    rep = config.params.exp.rep

    genco = Exp(model_id, dataset_name, pos)
    if model_id == "openai/clip-vit-base-patch32":
        model = 'clip'
    elif model_id == "kakaobrain/align-base":
        model = 'align'
    else: model = 'blip'

    for i in range(rep):
        result_df = experiment(model_id, dataset_name, pos)
        result_df.to_csv(f'./result/{model}_{dataset_name}_{pos}_{i}.csv')
        print(f"mean Precision@1 : {result_df['answer'].mean()}")


if __name__ == "__main__":
    rep_exp()