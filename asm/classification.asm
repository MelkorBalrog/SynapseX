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
OP_NEUR CONFIG_ANN 0 FINALIZE 0.2 trained_weights
OP_NEUR CONFIG_ANN 1 FINALIZE 0.2 trained_weights
OP_NEUR CONFIG_ANN 2 FINALIZE 0.2 trained_weights

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

; (D) Majority voting for all classes using loop based on hp.num_classes
ADDI $t0, $zero, 0             ; index for vote initialisation
init_votes:
BEQ $t0, $s0, votes_inited
SLL $t1, $t0, 2                ; offset = i * 4
SW $zero, votes($t1)
ADDI $t0, $t0, 1
J init_votes
votes_inited:

ADDI $t0, $zero, 0             ; iterate over ANN predictions
ADDI $t2, $zero, 3             ; number of ANNs (fixed)
count_loop:
BEQ $t0, $t2, count_done
SLL $t1, $t0, 2
LW $t3, ann_preds($t1)
SLL $t4, $t3, 2                ; vote index
LW $t5, votes($t4)
ADDI $t5, $t5, 1
SW $t5, votes($t4)
ADDI $t0, $t0, 1
J count_loop
count_done:

; (E) Select class with highest vote
ADDI $t0, $zero, 0             ; class index
ADDI $t6, $zero, -1            ; max vote count
ADDI $t9, $zero, 0             ; predicted class
max_loop:
BEQ $t0, $s0, finalize_output
SLL $t1, $t0, 2
LW $t3, votes($t1)
BGT $t3, $t6, update_max
ADDI $t0, $t0, 1
J max_loop
update_max:
ADD $t6, $t3, $zero
ADD $t9, $t0, $zero
ADDI $t0, $t0, 1
J max_loop

finalize_output:
; Final result is in $t9. Use green color when class 0 is predicted.
HALT

.data
votes:
    .space 64                   ; vote counts for each class (>= hp.num_classes)
ann_preds:
    .space 12                   ; argmax results from the three ANNs
