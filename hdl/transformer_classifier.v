`timescale 1ns/1ps
// Transformer-based classifier hardware.
// Implements a minimal inference pipeline matching the Python
// TransformerClassifier in synapsex/models.py. Only classification
// is supported; training logic is intentionally omitted.
module transformer_classifier #(
    parameter IMAGE_SIZE = 28,
    parameter NUM_CLASSES = 3,
    parameter PATCH_SIZE = IMAGE_SIZE/4
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire [7:0]        pixel_in,
    input  wire              pixel_valid,
    output reg               ready,
    output reg               done,
    output reg [1:0]         class_out
);
    localparam PATCHES    = (IMAGE_SIZE/PATCH_SIZE)*(IMAGE_SIZE/PATCH_SIZE);
    localparam EMBED_DIM  = PATCH_SIZE*PATCH_SIZE;

    // Internal memories for patch embeddings and transformer activations
    reg [15:0] patch_mem     [0:PATCHES-1];
    reg [15:0] transformer_mem [0:PATCHES-1];

    // Simple linear head weights (placeholders)
    reg [15:0] head_w0 [0:PATCHES-1];
    reg [15:0] head_w1 [0:PATCHES-1];
    reg [15:0] head_w2 [0:PATCHES-1];

    // State machine definitions
    localparam S_IDLE      = 2'b00;
    localparam S_PATCH     = 2'b01;
    localparam S_TRANSFORM = 2'b10;
    localparam S_HEAD      = 2'b11;

    reg [1:0] state;
    integer i;
    reg [15:0] patch_acc;
    reg [15:0] pixel_cnt;
    reg [15:0] patch_cnt;
    reg [31:0] sum0;
    reg [31:0] sum1;
    reg [31:0] sum2;

    // Reset and initialization
    initial begin
        ready      = 1'b1;
        done       = 1'b0;
        class_out  = 2'b0;
        state      = S_IDLE;
        pixel_cnt  = 0;
        patch_cnt  = 0;
        patch_acc  = 0;
        sum0       = 0;
        sum1       = 0;
        sum2       = 0;
        for (i = 0; i < PATCHES; i = i + 1) begin
            patch_mem[i] = 0;
            transformer_mem[i] = 0;
            head_w0[i] = 0;
            head_w1[i] = 0;
            head_w2[i] = 0;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            ready     <= 1'b1;
            done      <= 1'b0;
            class_out <= 2'b0;
            patch_acc <= 0;
            pixel_cnt <= 0;
            patch_cnt <= 0;
            sum0 <= 0;
            sum1 <= 0;
            sum2 <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done  <= 1'b0;
                    if (start) begin
                        ready <= 1'b0;
                        state <= S_PATCH;
                    end
                end
                S_PATCH: begin
                    if (pixel_valid) begin
                        patch_acc <= patch_acc + pixel_in;
                        pixel_cnt <= pixel_cnt + 1;
                        if (pixel_cnt == EMBED_DIM - 1) begin
                            patch_mem[patch_cnt] <= patch_acc + pixel_in;
                            pixel_cnt <= 0;
                            patch_acc <= 0;
                            patch_cnt <= patch_cnt + 1;
                            if (patch_cnt == PATCHES - 1) begin
                                patch_cnt <= 0;
                                state <= S_TRANSFORM;
                            end
                        end
                    end
                end
                S_TRANSFORM: begin
                    // Placeholder for transformer layers
                    for (i = 0; i < PATCHES; i = i + 1) begin
                        transformer_mem[i] <= patch_mem[i];
                    end
                    state <= S_HEAD;
                end
                S_HEAD: begin
                    // Simple linear head: pick class with largest sum
                    sum0 <= 0;
                    sum1 <= 0;
                    sum2 <= 0;
                    for (i = 0; i < PATCHES; i = i + 1) begin
                        sum0 <= sum0 + transformer_mem[i] * head_w0[i];
                        sum1 <= sum1 + transformer_mem[i] * head_w1[i];
                        sum2 <= sum2 + transformer_mem[i] * head_w2[i];
                    end
                    if (sum0 >= sum1 && sum0 >= sum2)
                        class_out <= 2'd0;
                    else if (sum1 >= sum0 && sum1 >= sum2)
                        class_out <= 2'd1;
                    else
                        class_out <= 2'd2;
                    done  <= 1'b1;
                    ready <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
