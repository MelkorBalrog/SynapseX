`timescale 1ns/1ps
// Transformer-based classifier hardware.
// Implements a minimal inference pipeline matching the Python
// TransformerClassifier in synapsex/models.py. Only classification
// is supported; training logic is intentionally omitted.
module transformer_classifier #(
    parameter IMAGE_SIZE = 28,
    parameter NUM_CLASSES = 3,
    parameter PATCH_SIZE = IMAGE_SIZE/4,
    parameter HEADS = 4,
    parameter string WQ_FILE = "",
    parameter string WK_FILE = "",
    parameter string WV_FILE = "",
    parameter string FF1_FILE = "",
    parameter string FF2_FILE = "",
    parameter string HEADW0_FILE = "",
    parameter string HEADW1_FILE = "",
    parameter string HEADW2_FILE = ""
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
    reg [15:0] patch_mem       [0:PATCHES-1];
    reg [15:0] transformer_mem [0:PATCHES-1];

    // Transformer weights
    reg [15:0] wq     [0:HEADS-1];
    reg [15:0] wk     [0:HEADS-1];
    reg [15:0] wv     [0:HEADS-1];
    reg [15:0] ff1    [0:1]; // [0]=weight, [1]=bias
    reg [15:0] ff2    [0:1]; // [0]=weight, [1]=bias

    wire [15:0] ff1_w = ff1[0];
    wire [15:0] ff1_b = ff1[1];
    wire [15:0] ff2_w = ff2[0];
    wire [15:0] ff2_b = ff2[1];

    // Simple linear head weights
    reg [15:0] head_w0 [0:PATCHES-1];
    reg [15:0] head_w1 [0:PATCHES-1];
    reg [15:0] head_w2 [0:PATCHES-1];

    // State machine definitions
    localparam S_IDLE      = 2'b00;
    localparam S_PATCH     = 2'b01;
    localparam S_TRANSFORM = 2'b10;
    localparam S_HEAD      = 2'b11;

    reg [1:0] state;
    integer i, j, h;
    reg [15:0] patch_acc;
    reg [15:0] pixel_cnt;
    reg [15:0] patch_cnt;
    reg [31:0] sum0;
    reg [31:0] sum1;
    reg [31:0] sum2;

    reg [31:0] q;
    reg [31:0] k;
    reg [31:0] v;
    reg [31:0] score;
    reg [31:0] total;
    reg [31:0] weighted;
    reg [31:0] head_out [0:HEADS-1];
    reg [31:0] mh_out;

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
        for (h = 0; h < HEADS; h = h + 1) begin
            wq[h] = 0;
            wk[h] = 0;
            wv[h] = 0;
            head_out[h] = 0;
        end
        ff1[0] = 1; ff1[1] = 0;
        ff2[0] = 1; ff2[1] = 0;

        if (WQ_FILE  != "") $readmemh(WQ_FILE,  wq);
        if (WK_FILE  != "") $readmemh(WK_FILE,  wk);
        if (WV_FILE  != "") $readmemh(WV_FILE,  wv);
        if (FF1_FILE != "") $readmemh(FF1_FILE, ff1);
        if (FF2_FILE != "") $readmemh(FF2_FILE, ff2);
        if (HEADW0_FILE != "") $readmemh(HEADW0_FILE, head_w0);
        if (HEADW1_FILE != "") $readmemh(HEADW1_FILE, head_w1);
        if (HEADW2_FILE != "") $readmemh(HEADW2_FILE, head_w2);
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
                    // Single-layer multi-head self-attention + feedforward
                    for (i = 0; i < PATCHES; i = i + 1) begin
                        for (h = 0; h < HEADS; h = h + 1) begin
                            q = patch_mem[i] * wq[h];
                            total = 0;
                            weighted = 0;
                            for (j = 0; j < PATCHES; j = j + 1) begin
                                k = patch_mem[j] * wk[h];
                                v = patch_mem[j] * wv[h];
                                score = q * k;
                                total = total + score;
                                weighted = weighted + score * v;
                            end
                            if (total != 0)
                                head_out[h] = weighted / total;
                            else
                                head_out[h] = 0;
                        end
                        mh_out = 0;
                        for (h = 0; h < HEADS; h = h + 1)
                            mh_out = mh_out + head_out[h];
                        transformer_mem[i] <= (mh_out * ff1_w + ff1_b) * ff2_w + ff2_b;
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
