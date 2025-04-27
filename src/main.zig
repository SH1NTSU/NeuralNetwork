const std = @import("std");

const input: f32 = 0.2;
const expected: f32 = 15;
var weight: f32 = 4;
var bias: f32 = 4;
const learning_rate: f32 = 0.1;

pub fn main() !void {
    const start_time = std.time.microTimestamp();
    var epoch: usize = 0;
    while (epoch < 10000) : (epoch += 1) {
        const result = activation(weight, bias, input);
        const eror = expected - result;

        const dcost_dpred = -2.0 * eror;
        const dpred_dz = result * (1 - result);
        const dz_dw = input;
        const dz_db = 1.0;

        const dcost_dw = dcost_dpred * dpred_dz * dz_dw;
        const dcost_db = dcost_dpred * dpred_dz * dz_db;

        weight -= learning_rate * dcost_dw;
        bias -= learning_rate * dcost_db;

        if (epoch % 10000 == 0) {
            std.debug.print("Epoch {}: result={}\n", .{ epoch, result });
        }
    }

    const final_result = activation(weight, bias, input);
    const end_time = std.time.microTimestamp();
    const total_time = end_time - start_time;
    std.debug.print("total time u: {}\n", .{total_time});
    std.debug.print("After training: result={}\n", .{final_result});
    std.debug.print("Trained weight={}, bias={}\n", .{ weight, bias });
}

fn activation(w: f32, b: f32, x: f32) f32 {
    const i: f32 = w * x + b;
    return 1 / (1 + std.math.exp(-i));
}
