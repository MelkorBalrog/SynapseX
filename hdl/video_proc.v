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
