// Copyright (C) 2025 Miguel Marina
// Author: Miguel Marina <karel.capek.robotics@gmail.com>
// LinkedIn: https://www.linkedin.com/in/progman32/
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

`timescale 1ns/1ps
// Verilog representation of the classification assembly program.
// Each $display statement mirrors one line in asm/classification.asm so
// developers can reference the instruction set and craft their own apps.
module classification_asm;
    initial begin
        $display("; (A) Initialize the 3 Principal ANNs (architecture loaded from weights)");
        $display("OP_NEUR CONFIG_ANN 0 FINALIZE 0.2");
        $display("OP_NEUR CONFIG_ANN 1 FINALIZE 0.2");
        $display("OP_NEUR CONFIG_ANN 2 FINALIZE 0.2");
        $display("; (B) Load All Trained Weights for Classification");
        $display("OP_NEUR LOAD_ALL trained_weights");
        $display("; (C) Perform Inference on the 3 Principal ANNs");
        $display("OP_NEUR INFER_ANN 0 true 10");
        $display("ADD $t0, $zero, $t9");
        $display("OP_NEUR INFER_ANN 1 true 10");
        $display("ADD $t1, $zero, $t9");
        $display("OP_NEUR INFER_ANN 2 true 10");
        $display("ADD $t2, $zero, $t9");
        $display("; (D) Manual Majority Voting Among the 3 ANNs");
        $display("ADDI $t10, $zero, 0");
        $display("ADDI $t11, $zero, 0");
        $display("ADDI $t12, $zero, 0");
        $display("BEQ $t0, $zero, ann0_isA");
        $display("ADDI $at, $zero, 1");
        $display("BEQ $t0, $at, ann0_isB");
        $display("J ann0_isU");
        $display("ann0_isA:");
        $display("ADDI $t10, $t10, 1");
        $display("J ann0_done");
        $display("ann0_isB:");
        $display("ADDI $t11, $t11, 1");
        $display("J ann0_done");
        $display("ann0_isU:");
        $display("ADDI $t12, $t12, 1");
        $display("J ann0_done");
        $display("ann0_done:");
        $display("BEQ $t1, $zero, ann1_isA");
        $display("ADDI $at, $zero, 1");
        $display("BEQ $t1, $at, ann1_isB");
        $display("J ann1_isU");
        $display("ann1_isA:");
        $display("ADDI $t10, $t10, 1");
        $display("J ann1_done");
        $display("ann1_isB:");
        $display("ADDI $t11, $t11, 1");
        $display("J ann1_done");
        $display("ann1_isU:");
        $display("ADDI $t12, $t12, 1");
        $display("J ann1_done");
        $display("ann1_done:");
        $display("BEQ $t2, $zero, ann2_isA");
        $display("ADDI $at, $zero, 1");
        $display("BEQ $t2, $at, ann2_isB");
        $display("J ann2_isU");
        $display("ann2_isA:");
        $display("ADDI $t10, $t10, 1");
        $display("J ann2_done");
        $display("ann2_isB:");
        $display("ADDI $t11, $t11, 1");
        $display("J ann2_done");
        $display("ann2_isU:");
        $display("ADDI $t12, $t12, 1");
        $display("J ann2_done");
        $display("ann2_done:");
        $display("; (E) Decide Final Class based on Majority Vote");
        $display("BGT $t10, $t11, checkA_vsU");
        $display("BGT $t11, $t10, checkB_vsU");
        $display("ADDI $t9, $zero, 2");
        $display("J finalize_output");
        $display("checkA_vsU:");
        $display("BGT $t10, $t12, finalizeA");
        $display("ADDI $t9, $zero, 2");
        $display("J finalize_output");
        $display("checkB_vsU:");
        $display("BGT $t11, $t12, finalizeB");
        $display("ADDI $t9, $zero, 2");
        $display("J finalize_output");
        $display("finalizeA:");
        $display("ADDI $t9, $zero, 0");
        $display("J finalize_output");
        $display("finalizeB:");
        $display("ADDI $t9, $zero, 1");
        $display("finalize_output:");
        $display("HALT");
    end
endmodule
