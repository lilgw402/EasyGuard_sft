from .modeling_deberta_moe import (
    DebertaV2ForMaskedLMMoE,
    DebertaV2ForQuestionAnsweringMoE, MLP,
    DebertaV2ForSequenceClassificationMoE,
    DebertaV2ForTokenClassificationMoE
)

from .modeling_xlm_roberta import (
    XLMRobertaForCausalLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification
)