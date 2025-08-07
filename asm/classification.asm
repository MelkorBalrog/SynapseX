; Copyright (C) 2025 Miguel Marina
; Author: Miguel Marina <karel.capek.robotics@gmail.com>
; LinkedIn: https://www.linkedin.com/in/progman32/
;
; This program is free software: you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation, either version 3 of the License, or
; (at your option) any later version.
;
; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.
;
; You should have received a copy of the GNU General Public License
; along with this program.  If not, see <https://www.gnu.org/licenses/>.

; (A) Initialize the 3 Principal ANNs (architecture loaded from weights)
OP_NEUR CONFIG_ANN 0 FINALIZE 0.2
OP_NEUR CONFIG_ANN 1 FINALIZE 0.2
OP_NEUR CONFIG_ANN 2 FINALIZE 0.2

; (A1) Load class count from configuration so assembly and Python agree
OP_NEUR GET_NUM_CLASSES
ADD $s0, $zero, $t9            ; $s0 holds hp.num_classes

; (B) Load All Trained Weights for Classification
OP_NEUR LOAD_ALL trained_weights

; (C) Perform Inference on the 3 Principal ANNs and fetch argmax directly
OP_NEUR INFER_ANN 0 true 10
OP_NEUR GET_ARGMAX 0
SW $t9, ann_preds              ; store ANN0 prediction
OP_NEUR INFER_ANN 1 true 10
OP_NEUR GET_ARGMAX 1
SW $t9, ann_preds+4            ; store ANN1 prediction
OP_NEUR INFER_ANN 2 true 10
OP_NEUR GET_ARGMAX 2
SW $t9, ann_preds+8            ; store ANN2 prediction

; (D) Majority voting computed by Neural IP
OP_NEUR MAJORITY_VOTE
HALT

.data
ann_preds:
    .space 12                   ; argmax results from the three ANNs
