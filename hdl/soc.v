`timescale 1ns/1ps
// Simple SoC integrating CPU, ALU, memories and the transformer classifier.
module soc(
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  pixel_in,
    input  wire        pixel_valid,
    output wire [1:0]  class_out,
    output wire        done
);
    // Program counter and register file (16 general purpose registers)
    reg [7:0]  pc;
    reg [31:0] regs [0:15];
    reg        running;

    // Instruction and data memories
    wire [31:0] instr;
    wire [31:0] data_r;
    reg  [31:0] data_w;
    reg         data_we;

    wishbone_memory #(.ADDR_WIDTH(8), .DATA_WIDTH(32)) instr_mem(
        .clk(clk),
        .addr(pc),
        .wdata(32'b0),
        .we(1'b0),
        .rdata(instr)
    );

    wishbone_memory #(.ADDR_WIDTH(8), .DATA_WIDTH(32)) data_mem(
        .clk(clk),
        .addr(regs[15][7:0]),
        .wdata(data_w),
        .we(data_we),
        .rdata(data_r)
    );

    // ALU wires
    reg  [3:0]  alu_op;
    reg  [31:0] alu_a;
    reg  [31:0] alu_b;
    wire [31:0] alu_res;
    wire        alu_zero;
    wire        alu_gt;

    alu alu0(
        .op(alu_op),
        .a(alu_a),
        .b(alu_b),
        .result(alu_res),
        .zero(alu_zero),
        .gt(alu_gt)
    );

    // Transformer classifier instance
    reg  cls_start;
    wire cls_ready;
    transformer_classifier classifier(
        .clk(clk),
        .rst(rst),
        .start(cls_start),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .ready(cls_ready),
        .done(done),
        .class_out(class_out)
    );

    // Opcode definitions
    localparam OP_NOP  = 8'h00;
    localparam OP_ADD  = 8'h01;
    localparam OP_ADDI = 8'h02;
    localparam OP_BEQ  = 8'h03;
    localparam OP_BGT  = 8'h04;
    localparam OP_J    = 8'h05;
    localparam OP_HALT = 8'h06;
    localparam OP_NEUR = 8'h07;

    // Instruction field extraction
    wire [7:0] opcode = instr[31:24];
    wire [7:0] rd     = instr[23:16];
    wire [7:0] rs     = instr[15:8];
    wire [7:0] rtimm  = instr[7:0];

    // Example program: run classifier once then halt
    initial begin
        pc      = 0;
        running = 1'b1;
        cls_start = 1'b0;
        instr_mem.mem[0] = {OP_NEUR, 8'd9, 16'd0};
        instr_mem.mem[1] = {OP_HALT, 24'd0};
    end

    // CPU execution loop
    always @(posedge clk) begin
        if (rst) begin
            pc       <= 0;
            running  <= 1'b1;
            cls_start <= 1'b0;
        end else if (running) begin
            case (opcode)
                OP_NOP: pc <= pc + 1;
                OP_ADD: begin
                    regs[rd[3:0]] <= regs[rs[3:0]] + regs[rtimm[3:0]];
                    pc <= pc + 1;
                end
                OP_ADDI: begin
                    regs[rd[3:0]] <= regs[rs[3:0]] + {{24{rtimm[7]}}, rtimm};
                    pc <= pc + 1;
                end
                OP_BEQ: begin
                    if (regs[rd[3:0]] == regs[rs[3:0]])
                        pc <= pc + {{24{rtimm[7]}}, rtimm};
                    else
                        pc <= pc + 1;
                end
                OP_BGT: begin
                    if (regs[rd[3:0]] > regs[rs[3:0]])
                        pc <= pc + {{24{rtimm[7]}}, rtimm};
                    else
                        pc <= pc + 1;
                end
                OP_J: pc <= {rs, rtimm};
                OP_NEUR: begin
                    if (cls_ready) begin
                        cls_start <= 1'b1;
                    end
                    if (done) begin
                        cls_start <= 1'b0;
                        regs[rd[3:0]] <= {30'b0, class_out};
                        pc <= pc + 1;
                    end
                end
                OP_HALT: running <= 1'b0;
                default: pc <= pc + 1;
            endcase
        end
    end
endmodule
