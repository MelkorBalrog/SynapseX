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
// Minimal PCIe bridge placeholder mirroring the Python PCIeBridge class.
module pcie_bridge(
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] wdata,
    input  wire        we,
    output wire [31:0] rdata
);
    reg [31:0] reg0;
    always @(posedge clk) begin
        if (rst)
            reg0 <= 32'b0;
        else if (we)
            reg0 <= wdata;
    end
    assign rdata = reg0;
endmodule
