`timescale 1ns/1ps
module transformer_classifier_tb;
    localparam IMAGE_SIZE = 8;
    localparam PATCH_SIZE = IMAGE_SIZE/4;
    localparam TOTAL_PIXELS = IMAGE_SIZE*IMAGE_SIZE;
    reg clk = 0;
    reg rst = 1;
    reg start = 0;
    reg pixel_valid = 0;
    reg [7:0] pixel_in = 0;
    wire ready;
    wire done;
    wire [1:0] class_out;

    transformer_classifier #(
        .IMAGE_SIZE(IMAGE_SIZE),
        .NUM_CLASSES(3),
        .PATCH_SIZE(PATCH_SIZE),
        .HEADS(1),
        .WQ_FILE("hdl/testdata/wq.mem"),
        .WK_FILE("hdl/testdata/wk.mem"),
        .WV_FILE("hdl/testdata/wv.mem"),
        .FF1_FILE("hdl/testdata/ff1.mem"),
        .FF2_FILE("hdl/testdata/ff2.mem"),
        .HEADW0_FILE("hdl/testdata/head_w0.mem"),
        .HEADW1_FILE("hdl/testdata/head_w1.mem"),
        .HEADW2_FILE("hdl/testdata/head_w2.mem")
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .ready(ready),
        .done(done),
        .class_out(class_out)
    );

    reg [7:0] pixels [0:TOTAL_PIXELS-1];
    integer i;

    always #5 clk = ~clk;

    initial begin
        $readmemh("hdl/testdata/pixels.mem", pixels);
        #20 rst = 0;
        @(posedge clk);
        start = 1;
        pixel_valid = 1;
        for (i = 0; i < TOTAL_PIXELS; i = i + 1) begin
            pixel_in = pixels[i];
            @(posedge clk);
        end
        pixel_valid = 0;
        start = 0;
        wait(done);
        $display("class_out=%0d", class_out);
        if (class_out == 0)
            $display("TEST PASSED");
        else
            $display("TEST FAILED");
        $finish;
    end
endmodule
