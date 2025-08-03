; (A) Define the 4 Principal ANNs with Transformer Layers
OP_NEUR CONFIG_ANN 0 CREATE_LAYER 784 2352 LEAKYRELU
OP_NEUR CONFIG_ANN 0 SET_SHAPE 28 28 1
OP_NEUR CONFIG_ANN 0 ADD_LAYER 2352 TRANSFORMER
OP_NEUR CONFIG_ANN 0 ADD_LAYER 2352 TRANSFORMER
OP_NEUR CONFIG_ANN 0 ADD_LAYER 2352 TRANSFORMER
OP_NEUR CONFIG_ANN 0 ADD_LAYER 2352 LEAKYRELU
OP_NEUR CONFIG_ANN 0 ADD_LAYER 3 NONE
OP_NEUR CONFIG_ANN 0 FINALIZE
OP_NEUR CONFIG_ANN 1 CREATE_LAYER 784 256 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 2560 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 256 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 2560 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 256 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 2560 LEAKYRELU
OP_NEUR CONFIG_ANN 1 ADD_LAYER 3 NONE
OP_NEUR CONFIG_ANN 1 FINALIZE
OP_NEUR CONFIG_ANN 2 CREATE_LAYER 784 256 LEAKYRELU
OP_NEUR CONFIG_ANN 2 ADD_LAYER 9256 COMBINED_STEP
OP_NEUR CONFIG_ANN 2 ADD_LAYER 256 LEAKYRELU
OP_NEUR CONFIG_ANN 2 ADD_LAYER 9256 COMBINED_STEP
OP_NEUR CONFIG_ANN 2 ADD_LAYER 256 LEAKYRELU
OP_NEUR CONFIG_ANN 2 ADD_LAYER 9256 COMBINED_STEP
OP_NEUR CONFIG_ANN 2 ADD_LAYER 3 NONE
OP_NEUR CONFIG_ANN 2 FINALIZE
; (B) Load All Trained Weights for Classification
OP_NEUR LOAD_ALL trained_weights
; (C) Perform Inference on the 4 Principal ANNs
OP_NEUR INFER_ANN 0 true 10
ADD $t0, $zero, $t9
OP_NEUR INFER_ANN 1 true 10
ADD $t1, $zero, $t9
OP_NEUR INFER_ANN 2 true 10
ADD $t2, $zero, $t9
; (D) Manual Majority Voting Among the 4 ANNs
ADDI $t10, $zero, 0
ADDI $t11, $zero, 0
ADDI $t12, $zero, 0
BEQ $t0, $zero, ann0_isA
ADDI $at, $zero, 1
BEQ $t0, $at, ann0_isB
J ann0_isU
ann0_isA:
ADDI $t10, $t10, 1
J ann0_done
ann0_isB:
ADDI $t11, $t11, 1
J ann0_done
ann0_isU:
ADDI $t12, $t12, 1
J ann0_done
ann0_done:
BEQ $t1, $zero, ann1_isA
ADDI $at, $zero, 1
BEQ $t1, $at, ann1_isB
J ann1_isU
ann1_isA:
ADDI $t10, $t10, 1
J ann1_done
ann1_isB:
ADDI $t11, $t11, 1
J ann1_done
ann1_isU:
ADDI $t12, $t12, 1
J ann1_done
ann1_done:
BEQ $t2, $zero, ann2_isA
ADDI $at, $zero, 1
BEQ $t2, $at, ann2_isB
J ann2_isU
ann2_isA:
ADDI $t10, $t10, 1
J ann2_done
ann2_isB:
ADDI $t11, $t11, 1
J ann2_done
ann2_isU:
ADDI $t12, $t12, 1
J ann2_done
ann2_done:
; (E) Decide Final Class based on Majority Vote
BGT $t10, $t11, checkA_vsU
BGT $t11, $t10, checkB_vsU
ADDI $t9, $zero, 2
J finalize_output
checkA_vsU:
BGT $t10, $t12, finalizeA
ADDI $t9, $zero, 2
J finalize_output
checkB_vsU:
BGT $t11, $t12, finalizeB
ADDI $t9, $zero, 2
J finalize_output
finalizeA:
ADDI $t9, $zero, 0
J finalize_output
finalizeB:
ADDI $t9, $zero, 1
finalize_output:
HALT
