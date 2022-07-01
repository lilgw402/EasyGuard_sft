# -*- coding: utf-8 -*-
from ..text_ops.tokenization import TaskManager

flag = True
try:
    import matx
    if matx.__version__.startswith("1.5"):
        from ..text_ops.tokenization import BertTokenizer
        from ..text_ops.multi_domain_concat import MultiDomainConcatBuilder
        from ..text_ops.bert_input_builder import BertInputsBuilder, BertQueryStackOnDocsInputsBuilder
        from ..text_ops.embedding_process import EmbedProcess, EmbedProcessMatx
        from ..text_ops.matx_text_pipe import MatxTextBasePipe
    else:
        flag = False
except Exception:
    flag = False
    print("[NOTICE] Matx found in FEX/Matx Ops, please check !")
if not flag:
    BertTokenizer = None
    MultiDomainConcatBuilder = None
    BertInputsBuilder = None
    BertQueryStackOnDocsInputsBuilder = None
    EmbedProcess = None
    EmbedProcessMatx = None
    MatxTextBasePipe = None
