// Limen safety supervisor reference RTL.
// Reference block only: not synthesized, formally verified, or certified.
module limen_safety_supervisor #(
    parameter integer T_HIGH_PERMILLE                  = 750,
    parameter integer T_LOW_PERMILLE                   = 450,
    parameter integer PERSISTENCE_WINDOWS              = 3,
    parameter integer HARD_MAX_TORQUE_MNM              = 40000,
    parameter integer HARD_MAX_VELOCITY_URAD_S         = 6000000,
    parameter integer DEGRADED_TORQUE_SCALE_PERMILLE   = 400,
    parameter integer DEGRADED_VELOCITY_SCALE_PERMILLE = 500
) (
    input  wire        clk,
    input  wire        reset_n,
    input  wire        valid,
    input  wire [15:0] confidence_permille,
    input  wire [15:0] reliability_permille,
    input  wire signed [31:0] requested_torque_mnm,
    input  wire signed [31:0] requested_velocity_urad_s,
    output reg  [1:0]  state,
    output reg  signed [31:0] torque_mnm,
    output reg  signed [31:0] velocity_urad_s,
    output reg  [31:0] effective_confidence_permille,
    output reg         supervisor_veto
);

    localparam [1:0] STATE_SAFE_HALT = 2'd0;
    localparam [1:0] STATE_DEGRADED  = 2'd1;
    localparam [1:0] STATE_NOMINAL   = 2'd2;

    reg [31:0] confidence_product;
    reg [31:0] effective_next;
    reg [31:0] low_streak;
    reg [31:0] low_streak_next;
    reg [1:0]  state_next;
    reg signed [31:0] torque_next;
    reg signed [31:0] velocity_next;
    reg veto_next;

    always @* begin
        confidence_product = confidence_permille * reliability_permille;
        effective_next     = confidence_product / 1000;
        low_streak_next    = low_streak;
        state_next         = state;
        torque_next        = 32'sd0;
        velocity_next      = 32'sd0;
        veto_next          = 1'b0;

        if (!valid || (confidence_permille > 16'd1000) || (reliability_permille > 16'd1000)) begin
            state_next      = STATE_SAFE_HALT;
            low_streak_next = 0;
        end else if (effective_next < T_LOW_PERMILLE) begin
            low_streak_next = low_streak + 1;
            if (low_streak_next >= PERSISTENCE_WINDOWS)
                state_next = STATE_SAFE_HALT;
            else
                state_next = STATE_DEGRADED;
        end else begin
            low_streak_next = 0;
            if (effective_next >= T_HIGH_PERMILLE)
                state_next = STATE_NOMINAL;
            else
                state_next = STATE_DEGRADED;
        end

        if (state_next == STATE_NOMINAL) begin
            torque_next   = requested_torque_mnm;
            velocity_next = requested_velocity_urad_s;
        end else if (state_next == STATE_DEGRADED) begin
            torque_next   = (requested_torque_mnm   * DEGRADED_TORQUE_SCALE_PERMILLE)   / 1000;
            velocity_next = (requested_velocity_urad_s * DEGRADED_VELOCITY_SCALE_PERMILLE) / 1000;
        end

        if (torque_next > HARD_MAX_TORQUE_MNM) begin
            torque_next = HARD_MAX_TORQUE_MNM;
            veto_next   = 1'b1;
        end else if (torque_next < -HARD_MAX_TORQUE_MNM) begin
            torque_next = -HARD_MAX_TORQUE_MNM;
            veto_next   = 1'b1;
        end
        if (velocity_next > HARD_MAX_VELOCITY_URAD_S) begin
            velocity_next = HARD_MAX_VELOCITY_URAD_S;
            veto_next     = 1'b1;
        end else if (velocity_next < -HARD_MAX_VELOCITY_URAD_S) begin
            velocity_next = -HARD_MAX_VELOCITY_URAD_S;
            veto_next     = 1'b1;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state                          <= STATE_SAFE_HALT;
            low_streak                     <= 0;
            torque_mnm                     <= 0;
            velocity_urad_s                <= 0;
            effective_confidence_permille  <= 0;
            supervisor_veto                <= 1'b0;
        end else begin
            state                          <= state_next;
            low_streak                     <= low_streak_next;
            torque_mnm                     <= torque_next;
            velocity_urad_s                <= velocity_next;
            effective_confidence_permille  <= effective_next;
            supervisor_veto                <= veto_next;
        end
    end
endmodule
