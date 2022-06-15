
import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        '''NOTE: You need to finish the code of computing f1-score.

        '''
        get_num = 0
        sum_num = 0
        
        pred = predictions[:, :]
        labels = labels[:, :]        
        pred_entities = extract_entities(pred, False, True)
        true_entities = extract_entities(labels, False, True)

        for pre, tru in zip(pred_entities, true_entities):
            pred_entities = set(pre)
            true_entities = set(tru)
            get_num += len(pred_entities & true_entities)
            sum_num += len(pred_entities) + len(true_entities)

        f1_score = 2 * get_num / sum_num + 1e-8

        return {"f1": f1_score}


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        '''NOTE: You need to finish the code of computing f1-score.

        '''
        get_num = 0
        sum_num = 0
        
        pred1 = predictions[:, :, 0]
        labels1 = labels1[:, :]
        pred_entities1 = extract_entities(pred1, True, True)
        true_entities1 = extract_entities(labels1, True, True)

        for pre1, tru1 in zip(pred_entities1, true_entities1):
            pred_entities = set(pre1)
            true_entities = set(tru1)
            get_num += len(pred_entities & true_entities)
            sum_num += len(pred_entities) + len(true_entities)   
            
        pred2 = predictions[:, :, 1]
        labels2 = labels2[:, :]            
        pred_entities2 = extract_entities(pred2, True, False)   
        true_entities2 = extract_entities(labels2, True, False) 
        
        for pre2, tru2 in zip(pred_entities2, true_entities2):
            pred_entities = set(pre2)
            true_entities = set(tru2)
            get_num += len(pred_entities & true_entities)
            sum_num += len(pred_entities) + len(true_entities) 
        
        f1_score = 2 * get_num / sum_num + 1e-8

        return {"f1": f1_score}


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.

    '''

    def _choose_entity_type(_types: List[str]) -> str:
        _pairs = sorted(Counter(_types).items(), key=lambda _pair: (_pair[1], _LABEL_RANK[_pair[0]]), reverse=True)
        _type = _pairs[0][0]
        return _type

    for batch in batch_labels_or_preds.tolist():
        entities = []  # e: (start_idx, end_idx, type)
        start_idx, end_idx = -1, -1
        types = []
        if_end = False

        for i, _label in enumerate(batch):
            label = id2label[_label]

            if label[0] == "B":
                if len(types):
                    entities.append((start_idx, end_idx, _choose_entity_type(types)))
                    types.clear()

                if_end = True
                end_idx = start_idx = i
                types.append(label[2:])
                
            elif (label[0] == "I") and (if_end is True):
                end_idx = i
                types.append(label[2:])

            else:
                if len(types):
                    entities.append((start_idx, end_idx, _choose_entity_type(types)))
                    types.clear()

                if_end = False
                types.clear()

        if len(types):
            entities.append((start_idx, end_idx, _choose_entity_type(types)))

        batch_entities.append(entities)
    
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    