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
; (C) Perform Inference on the 3 Principal ANNs
OP_NEUR INFER_ANN 0 true 10
ADD $t0, $zero, $t9
OP_NEUR INFER_ANN 1 true 10
ADD $t1, $zero, $t9
OP_NEUR INFER_ANN 2 true 10
ADD $t2, $zero, $t9
; (D) Naive Bayesian inference combining ANN outputs
; Initialize log-likelihood accumulators for classes A, B, and U
ADDI $t10, $zero, 0
ADDI $t11, $zero, 0
ADDI $t12, $zero, 0
; Preload log-probability constants (scaled by 1000)
ADDI $t13, $zero, -105      ; log(0.9)  ≈ -0.105
ADDI $t14, $zero, -2995     ; log(0.05) ≈ -2.995

; Process ANN0 prediction
BEQ $t0, $zero, ann0_isA
ADDI $at, $zero, 1
BEQ $t0, $at, ann0_isB
J ann0_isU
ann0_isA:
ADD $t10, $t10, $t13
ADD $t11, $t11, $t14
ADD $t12, $t12, $t14
J ann0_done
ann0_isB:
ADD $t11, $t11, $t13
ADD $t10, $t10, $t14
ADD $t12, $t12, $t14
J ann0_done
ann0_isU:
ADD $t12, $t12, $t13
ADD $t10, $t10, $t14
ADD $t11, $t11, $t14
ann0_done:

; Process ANN1 prediction
BEQ $t1, $zero, ann1_isA
ADDI $at, $zero, 1
BEQ $t1, $at, ann1_isB
J ann1_isU
ann1_isA:
ADD $t10, $t10, $t13
ADD $t11, $t11, $t14
ADD $t12, $t12, $t14
J ann1_done
ann1_isB:
ADD $t11, $t11, $t13
ADD $t10, $t10, $t14
ADD $t12, $t12, $t14
J ann1_done
ann1_isU:
ADD $t12, $t12, $t13
ADD $t10, $t10, $t14
ADD $t11, $t11, $t14
ann1_done:

; Process ANN2 prediction
BEQ $t2, $zero, ann2_isA
ADDI $at, $zero, 1
BEQ $t2, $at, ann2_isB
J ann2_isU
ann2_isA:
ADD $t10, $t10, $t13
ADD $t11, $t11, $t14
ADD $t12, $t12, $t14
J ann2_done
ann2_isB:
ADD $t11, $t11, $t13
ADD $t10, $t10, $t14
ADD $t12, $t12, $t14
J ann2_done
ann2_isU:
ADD $t12, $t12, $t13
ADD $t10, $t10, $t14
ADD $t11, $t11, $t14
ann2_done:

; (E) Decide Final Class based on Maximum Posterior
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
ADDI $t9, $zero, 0           ; Class A selected – display a green indicator
J finalize_output
finalizeB:
ADDI $t9, $zero, 1           ; Class B selected – no green indicator
finalize_output:
; Final result is in $t9. Use green color when class A is predicted.
HALT
