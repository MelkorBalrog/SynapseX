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

; (A) Initialize the 3 Principal ANNs with default dropout
OP_NEUR CONFIG_ANN 0 FINALIZE 0.2
OP_NEUR CONFIG_ANN 1 FINALIZE 0.2
OP_NEUR CONFIG_ANN 2 FINALIZE 0.2

; (B) Tune hyperparameters via a genetic algorithm
OP_NEUR TUNE_GA 0 5 8
OP_NEUR TUNE_GA 1 5 8
OP_NEUR TUNE_GA 2 5 8

; (C) Train Principal ANNs using the tuned parameters
OP_NEUR TRAIN_ANN 0 40 0.005 16
OP_NEUR TRAIN_ANN 1 40 0.005 16
OP_NEUR TRAIN_ANN 2 40 0.005 16

; (D) Save all weights
OP_NEUR SAVE_ALL trained_weights

; Training complete – display a green indicator to confirm success
HALT
