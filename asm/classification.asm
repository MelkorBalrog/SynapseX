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
; (B) Load All Trained Weights for Classification
OP_NEUR LOAD_ALL trained_weights
; (C) Perform Inference on the 3 Principal ANNs
OP_NEUR INFER_ANN 0 true 10
ADD $t0, $zero, $t9
OP_NEUR INFER_ANN 1 true 10
ADD $t1, $zero, $t9
OP_NEUR INFER_ANN 2 true 10
ADD $t2, $zero, $t9
; (D) Manual Majority Voting Among the 3 ANNs
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
ADDI $t9, $zero, 0           ; Class A selected – display a green indicator
J finalize_output
finalizeB:
ADDI $t9, $zero, 1           ; Class B selected – no green indicator
finalize_output:
; Final result is in $t9. Use green color when class A is predicted.
HALT
