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
// Simple video processing IP block mirroring the Python VideoProcIP class.
// When started it reads a word from memory, adds 0x100, writes the result
// back and raises 'done'.
module video_proc(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [31:0] in_addr,
    input  wire [31:0] out_addr,
    output reg         done,
    // memory interface
    output reg  [31:0] mem_addr,
    input  wire [31:0] mem_rdata,
    output reg  [31:0] mem_wdata,
    output reg         mem_we
);
    localparam IDLE  = 2'd0,
               READ  = 2'd1,
               WRITE = 2'd2,
               DONE  = 2'd3;
    reg [1:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state    <= IDLE;
            done     <= 1'b0;
            mem_we   <= 1'b0;
            mem_addr <= 32'b0;
            mem_wdata<= 32'b0;
        end else begin
            case (state)
                IDLE: begin
                    done   <= 1'b0;
                    mem_we <= 1'b0;
                    if (start) begin
                        mem_addr <= in_addr;
                        state    <= READ;
                    end
                end
                READ: begin
                    mem_wdata <= mem_rdata + 32'h100;
                    mem_addr  <= out_addr;
                    mem_we    <= 1'b1;
                    state     <= WRITE;
                end
                WRITE: begin
                    mem_we <= 1'b0;
                    state  <= DONE;
                end
                DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= IDLE;
                end
            endcase
        end
    end
endmodule
