// Limen Hardware Event Queue
// BRAM-backed FIFO event queue with timestamp tracking and overflow protection.
`timescale 1ns / 1ps

module limen_event_queue #(
    parameter integer DEPTH        = 256,
    parameter integer CHANNEL_ID_W = 4,
    parameter integer TS_WIDTH     = 32,
    parameter integer PTR_W        = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Ingest push interface
    input  wire                    push_valid,
    input  wire [CHANNEL_ID_W-1:0] push_channel_id,
    input  wire [TS_WIDTH-1:0]     push_timestamp,
    output wire                    push_ready,

    // Consumer pop interface
    input  wire                    pop_req,
    output reg                     pop_valid,
    output reg  [CHANNEL_ID_W-1:0] pop_channel_id,
    output reg  [TS_WIDTH-1:0]     pop_timestamp,

    output wire [PTR_W:0]          queue_count,
    output reg                     overflow_flag
);

    reg [CHANNEL_ID_W-1:0] id_mem [0:DEPTH-1];
    reg [TS_WIDTH-1:0]     ts_mem [0:DEPTH-1];

    reg [PTR_W-1:0] head_ptr;
    reg [PTR_W-1:0] tail_ptr;
    reg [PTR_W:0]   count;

    wire full  = (count == DEPTH);
    wire empty = (count == 0);
    assign push_ready  = !full;
    assign queue_count = count;

    wire do_push = push_valid && push_ready;
    wire do_pop  = pop_req && !empty;

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            id_mem[i] = {CHANNEL_ID_W{1'b0}};
            ts_mem[i] = {TS_WIDTH{1'b0}};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head_ptr       <= {PTR_W{1'b0}};
            tail_ptr       <= {PTR_W{1'b0}};
            count          <= {(PTR_W+1){1'b0}};
            pop_valid      <= 1'b0;
            pop_channel_id <= {CHANNEL_ID_W{1'b0}};
            pop_timestamp  <= {TS_WIDTH{1'b0}};
            overflow_flag  <= 1'b0;
        end else begin
            pop_valid <= 1'b0;

            if (push_valid && full) begin
                overflow_flag <= 1'b1;
            end

            if (do_push) begin
                id_mem[tail_ptr] <= push_channel_id;
                ts_mem[tail_ptr] <= push_timestamp;
                if (tail_ptr == DEPTH - 1)
                    tail_ptr <= {PTR_W{1'b0}};
                else
                    tail_ptr <= tail_ptr + 1'b1;
            end

            if (do_pop) begin
                pop_valid      <= 1'b1;
                pop_channel_id <= id_mem[head_ptr];
                pop_timestamp  <= ts_mem[head_ptr];
                if (head_ptr == DEPTH - 1)
                    head_ptr <= {PTR_W{1'b0}};
                else
                    head_ptr <= head_ptr + 1'b1;
            end

            case ({do_push, do_pop})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule
