#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <franka/exception.h>
#include <franka/robot.h>

namespace {

constexpr const char* kApprovalPhrase = "APPROVE PLUG-ZED CONTACT ENVELOPE AND MICRO-MOTION LIVE FR3";
constexpr const char* kReadyStatus = "PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY_READY";
constexpr const char* kReadyNotExecutedStatus =
    "PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY_READY_RUNTIME_NOT_EXECUTED";
constexpr const char* kExecutedPendingReviewStatus =
    "PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY_EXECUTED_PENDING_POSTFLIGHT_REVIEW";
constexpr const char* kBlockStatus = "BLOCK_PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY";
constexpr const char* kFutureAcceptStatus = "PLUG_ZED_CONTACT_MICRO_MOTION_RUNTIME_PROVEN";
constexpr const char* kFixtureContactStatus = "PLUG_FIXTURE_CONTACT_ENVELOPE_PROVEN";
constexpr const char* kLiveMicroMotionStatus = "FR3_LIVE_MICRO_MOTION_PROVEN";
constexpr const char* kObservationSchemaHash = "plugzedobs:v1:zed_left_zed_right:fci_gripper_state";
constexpr const char* kActionSchemaHash = "hilserl7d:v1:cartesian_delta_rpy_gripper";
constexpr const char* kDefaultResultRoot = "/sda/fzt/hilserl-fr3/plug_zed_insertion/contact_micro_motion";

struct Options {
  bool self_check = false;
  bool execute_live = false;
  bool confirm_human_physical_approval = false;
  bool confirm_estop_supervised = false;
  bool confirm_workspace_clear = false;
  bool confirm_robot_idle_initial_pose = false;
  bool confirm_fixture_staged = false;
  bool fixture_axis_base_confirmed = false;
  std::string approval_phrase;
  std::string robot_hostname = "172.16.0.2";
  std::string run_id = "plug_zed_contact_micro_motion_live_control_preview";
  std::string result_root = kDefaultResultRoot;
  double axis_x = 0.0;
  double axis_y = 0.0;
  double axis_z = 0.0;
  double max_contact_force_n = 5.0;
  double contact_force_limit_n = 8.0;
  double lateral_tolerance_m = 0.003;
  double insertion_depth_limit_m = 0.003;
  double retreat_distance_m = 0.020;
  double max_translation_step_m = 0.001;
  double max_rotation_step_rad = 0.0;
  double velocity_m_s = 0.003;
  double max_approach_duration_s = 1.0;
  double max_retreat_duration_s = 8.0;
};

struct Metrics {
  bool fixture_contact_envelope_executed = false;
  bool live_micro_motion_executed = false;
  bool fci_state_used = false;
  bool motion_command = false;
  bool force_limit_abort = false;
  bool contact_threshold_seen = false;
  bool insertion_depth_reached = false;
  bool approach_timeout = false;
  bool retreat_executed = false;
  bool retreat_completed = false;
  double elapsed_s = 0.0;
  double max_force_norm_n = 0.0;
  double max_axis_force_abs_n = 0.0;
  double max_approach_depth_m = 0.0;
  double retreat_distance_m = 0.0;
  std::string stop_reason = "NOT_STARTED";
};

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
    }
  }
  return out.str();
}

std::string Bool(bool value) {
  return value ? "true" : "false";
}

[[noreturn]] void Emit(const std::string& payload, int exit_code) {
  std::cout << payload << std::endl;
  std::exit(exit_code);
}

std::string CommonJson(const Options& options,
                       bool ok,
                       const std::string& status,
                       const std::string& extra = "") {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"ok\": " << Bool(ok) << ",\n";
  out << "  \"status\": \"" << JsonEscape(status) << "\",\n";
  out << "  \"approval_phrase_required\": \"" << kApprovalPhrase << "\",\n";
  out << "  \"future_accept_status\": \"" << kFutureAcceptStatus << "\",\n";
  out << "  \"future_fixture_contact_envelope_status\": \"" << kFixtureContactStatus << "\",\n";
  out << "  \"future_live_micro_motion_status\": \"" << kLiveMicroMotionStatus << "\",\n";
  out << "  \"block_status\": \"" << kBlockStatus << "\",\n";
  out << "  \"observation_schema_hash\": \"" << kObservationSchemaHash << "\",\n";
  out << "  \"action_schema_hash\": \"" << kActionSchemaHash << "\",\n";
  out << "  \"robot_hostname\": \"" << JsonEscape(options.robot_hostname) << "\",\n";
  out << "  \"run_id\": \"" << JsonEscape(options.run_id) << "\",\n";
  out << "  \"result_root\": \"" << JsonEscape(options.result_root) << "\",\n";
  out << "  \"fixture_axis_base_confirmed\": " << Bool(options.fixture_axis_base_confirmed) << ",\n";
  out << "  \"fixture_axis_base\": [" << options.axis_x << ", " << options.axis_y << ", "
      << options.axis_z << "],\n";
  out << "  \"max_contact_force_n\": " << options.max_contact_force_n << ",\n";
  out << "  \"contact_force_limit_n\": " << options.contact_force_limit_n << ",\n";
  out << "  \"lateral_tolerance_m\": " << options.lateral_tolerance_m << ",\n";
  out << "  \"insertion_depth_limit_m\": " << options.insertion_depth_limit_m << ",\n";
  out << "  \"retreat_distance_m\": " << options.retreat_distance_m << ",\n";
  out << "  \"max_translation_step_m\": " << options.max_translation_step_m << ",\n";
  out << "  \"max_rotation_step_rad\": " << options.max_rotation_step_rad << ",\n";
  out << "  \"velocity_m_s\": " << options.velocity_m_s << ",\n";
  out << "  \"max_approach_duration_s\": " << options.max_approach_duration_s << ",\n";
  out << "  \"max_retreat_duration_s\": " << options.max_retreat_duration_s << ",\n";
  out << "  \"runtime_execution_performed\": false,\n";
  out << "  \"fixture_contact_envelope_executed\": false,\n";
  out << "  \"live_micro_motion_executed\": false,\n";
  out << "  \"fr3_connection\": false,\n";
  out << "  \"fci_state_read\": false,\n";
  out << "  \"motion_command\": false,\n";
  out << "  \"gripper_command\": false,\n";
  out << "  \"policy_rollout\": false,\n";
  out << "  \"remote_training\": false,\n";
  out << "  \"checkpoint_promotion\": false,\n";
  out << "  \"droid_mutation\": false,\n";
  out << "  \"thread_019e4faf_touched\": false,\n";
  out << "  \"openpi_dependency\": false,\n";
  out << "  \"safe_to_claim_100_percent\": false,\n";
  out << "  \"overall_goal_complete\": false";
  if (!extra.empty()) {
    out << ",\n" << extra;
  } else {
    out << "\n";
  }
  out << "}";
  return out.str();
}

[[noreturn]] void Reject(const Options& options,
                         const std::string& code,
                         const std::string& detail = "") {
  std::ostringstream extra;
  extra << "  \"reject_code\": \"" << JsonEscape(code) << "\"";
  if (!detail.empty()) {
    extra << ",\n  \"detail\": \"" << JsonEscape(detail) << "\"";
  } else {
    extra << "\n";
  }
  Emit(CommonJson(options, false, kBlockStatus, extra.str()), 64);
}

bool IsBroadApproval(const std::string& value) {
  return value == "批准一切权限" || value == "批准一切真机权限" || value == "全部都批准" ||
         value == "approve all" || value == "all approved";
}

double ParseDouble(const std::string& value, const std::string& flag) {
  std::size_t parsed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("BAD_DOUBLE:" + flag);
  }
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::runtime_error("BAD_DOUBLE:" + flag);
  }
  return result;
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    std::string arg = argv[index];
    auto take_value = [&](const std::string& flag) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("MISSING_VALUE:" + flag);
      }
      return argv[++index];
    };
    if (arg == "--self-check") {
      options.self_check = true;
    } else if (arg == "--execute-live") {
      options.execute_live = true;
    } else if (arg == "--confirm-human-physical-approval-for-this-run") {
      options.confirm_human_physical_approval = true;
    } else if (arg == "--confirm-estop-supervised") {
      options.confirm_estop_supervised = true;
    } else if (arg == "--confirm-workspace-clear") {
      options.confirm_workspace_clear = true;
    } else if (arg == "--confirm-robot-idle-initial-pose") {
      options.confirm_robot_idle_initial_pose = true;
    } else if (arg == "--confirm-fixture-staged") {
      options.confirm_fixture_staged = true;
    } else if (arg == "--fixture-axis-base-confirmed") {
      options.fixture_axis_base_confirmed = true;
    } else if (arg == "--approval-phrase") {
      options.approval_phrase = take_value(arg);
    } else if (arg == "--robot-hostname") {
      options.robot_hostname = take_value(arg);
    } else if (arg == "--run-id") {
      options.run_id = take_value(arg);
    } else if (arg == "--result-root") {
      options.result_root = take_value(arg);
    } else if (arg == "--fixture-axis-base-x") {
      options.axis_x = ParseDouble(take_value(arg), arg);
    } else if (arg == "--fixture-axis-base-y") {
      options.axis_y = ParseDouble(take_value(arg), arg);
    } else if (arg == "--fixture-axis-base-z") {
      options.axis_z = ParseDouble(take_value(arg), arg);
    } else if (arg == "--max-contact-force-n") {
      options.max_contact_force_n = ParseDouble(take_value(arg), arg);
    } else if (arg == "--contact-force-limit-n") {
      options.contact_force_limit_n = ParseDouble(take_value(arg), arg);
    } else if (arg == "--lateral-tolerance-m") {
      options.lateral_tolerance_m = ParseDouble(take_value(arg), arg);
    } else if (arg == "--insertion-depth-limit-m") {
      options.insertion_depth_limit_m = ParseDouble(take_value(arg), arg);
    } else if (arg == "--retreat-distance-m") {
      options.retreat_distance_m = ParseDouble(take_value(arg), arg);
    } else if (arg == "--max-translation-step-m") {
      options.max_translation_step_m = ParseDouble(take_value(arg), arg);
    } else if (arg == "--max-rotation-step-rad") {
      options.max_rotation_step_rad = ParseDouble(take_value(arg), arg);
    } else if (arg == "--velocity-m-s") {
      options.velocity_m_s = ParseDouble(take_value(arg), arg);
    } else if (arg == "--max-approach-duration-s") {
      options.max_approach_duration_s = ParseDouble(take_value(arg), arg);
    } else if (arg == "--max-retreat-duration-s") {
      options.max_retreat_duration_s = ParseDouble(take_value(arg), arg);
    } else {
      throw std::runtime_error("UNKNOWN_ARG:" + arg);
    }
  }
  return options;
}

void ValidateApprovalAndGates(const Options& options) {
  if (options.approval_phrase.empty()) {
    Reject(options, "MISSING_APPROVAL_PHRASE");
  }
  if (IsBroadApproval(options.approval_phrase)) {
    Reject(options, "BROAD_APPROVAL_REJECTED");
  }
  if (options.approval_phrase != kApprovalPhrase) {
    Reject(options, "APPROVAL_PHRASE_MISMATCH");
  }
  if (!options.confirm_human_physical_approval) {
    Reject(options, "LIVE_ROBOT_GATE_MISSING", "human_physical_approval_for_this_run");
  }
  if (!options.confirm_estop_supervised) {
    Reject(options, "LIVE_ROBOT_GATE_MISSING", "estop_supervised");
  }
  if (!options.confirm_workspace_clear) {
    Reject(options, "LIVE_ROBOT_GATE_MISSING", "workspace_clear");
  }
  if (!options.confirm_robot_idle_initial_pose) {
    Reject(options, "LIVE_ROBOT_GATE_MISSING", "robot_idle_initial_pose");
  }
  if (!options.confirm_fixture_staged) {
    Reject(options, "LIVE_ROBOT_GATE_MISSING", "fixture_staged");
  }
}

std::array<double, 3> NormalizedAxisOrReject(const Options& options) {
  if (!options.fixture_axis_base_confirmed) {
    Reject(options, "FIXTURE_AXIS_BASE_NOT_CONFIRMED");
  }
  const double norm =
      std::sqrt(options.axis_x * options.axis_x + options.axis_y * options.axis_y +
                options.axis_z * options.axis_z);
  if (!std::isfinite(norm) || norm < 0.999 || norm > 1.001) {
    Reject(options, "FIXTURE_AXIS_BASE_NOT_UNIT_LENGTH");
  }
  return {{options.axis_x / norm, options.axis_y / norm, options.axis_z / norm}};
}

void ValidateEnvelope(const Options& options) {
  if (!(0.0 < options.max_contact_force_n &&
        options.max_contact_force_n <= options.contact_force_limit_n &&
        options.contact_force_limit_n <= 8.0)) {
    Reject(options, "CONTACT_FORCE_LIMIT_EXCEEDED");
  }
  if (options.max_contact_force_n > 5.0) {
    Reject(options, "MAX_CONTACT_FORCE_TOO_LARGE");
  }
  if (!(0.0 < options.lateral_tolerance_m && options.lateral_tolerance_m <= 0.003)) {
    Reject(options, "LATERAL_TOLERANCE_TOO_LARGE");
  }
  if (!(0.0 < options.insertion_depth_limit_m && options.insertion_depth_limit_m <= 0.003)) {
    Reject(options, "INSERTION_DEPTH_TOO_LARGE");
  }
  if (options.retreat_distance_m < 0.020) {
    Reject(options, "RETREAT_DISTANCE_TOO_SMALL");
  }
  if (!(0.0 < options.max_translation_step_m && options.max_translation_step_m <= 0.001)) {
    Reject(options, "TRANSLATION_STEP_TOO_LARGE");
  }
  if (options.max_rotation_step_rad != 0.0) {
    Reject(options, "ROTATION_STEP_NOT_ALLOWED");
  }
  if (!(0.0 < options.velocity_m_s && options.velocity_m_s <= 0.003)) {
    Reject(options, "VELOCITY_TOO_LARGE");
  }
  if (!(0.0 < options.max_approach_duration_s && options.max_approach_duration_s <= 1.0)) {
    Reject(options, "APPROACH_DURATION_TOO_LARGE");
  }
  const double minimum_retreat_time = options.retreat_distance_m / options.velocity_m_s;
  if (options.max_retreat_duration_s < minimum_retreat_time ||
      options.max_retreat_duration_s > 8.0) {
    Reject(options, "RETREAT_DURATION_OUT_OF_RANGE");
  }
}

double AxisProjection(const std::array<double, 3>& axis,
                      const std::array<double, 16>& pose,
                      const std::array<double, 16>& initial_pose) {
  return axis[0] * (pose[12] - initial_pose[12]) + axis[1] * (pose[13] - initial_pose[13]) +
         axis[2] * (pose[14] - initial_pose[14]);
}

double ForceNorm(const franka::RobotState& state) {
  return std::sqrt(state.O_F_ext_hat_K[0] * state.O_F_ext_hat_K[0] +
                   state.O_F_ext_hat_K[1] * state.O_F_ext_hat_K[1] +
                   state.O_F_ext_hat_K[2] * state.O_F_ext_hat_K[2]);
}

double AxisForceAbs(const franka::RobotState& state, const std::array<double, 3>& axis) {
  return std::abs(axis[0] * state.O_F_ext_hat_K[0] + axis[1] * state.O_F_ext_hat_K[1] +
                  axis[2] * state.O_F_ext_hat_K[2]);
}

Metrics ExecuteLive(const Options& options, const std::array<double, 3>& axis) {
  Metrics metrics;
  franka::Robot robot(options.robot_hostname);
  metrics.fci_state_used = true;

  robot.setCollisionBehavior(
      {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0}}, {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0}},
      {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0}}, {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0}},
      {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0}}, {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0}},
      {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0}}, {{8.0, 8.0, 8.0, 8.0, 8.0, 8.0}});

  const franka::RobotState initial_state = robot.readOnce();
  if (initial_state.robot_mode != franka::RobotMode::kIdle) {
    throw std::runtime_error("ROBOT_NOT_IDLE_AT_INITIAL_READ");
  }
  const std::array<double, 16> initial_pose = initial_state.O_T_EE;
  bool retreat_phase = false;
  double retreat_start_projection = 0.0;

  robot.control([&](const franka::RobotState& state,
                    franka::Duration period) -> franka::CartesianVelocities {
    metrics.elapsed_s += period.toSec();
    metrics.fixture_contact_envelope_executed = true;
    metrics.live_micro_motion_executed = true;
    metrics.motion_command = true;

    const double force_norm = ForceNorm(state);
    const double axis_force = AxisForceAbs(state, axis);
    metrics.max_force_norm_n = std::max(metrics.max_force_norm_n, force_norm);
    metrics.max_axis_force_abs_n = std::max(metrics.max_axis_force_abs_n, axis_force);

    const double projection = AxisProjection(axis, state.O_T_EE, initial_pose);
    metrics.max_approach_depth_m = std::max(metrics.max_approach_depth_m, projection);

    if (force_norm > options.contact_force_limit_n || axis_force > options.contact_force_limit_n) {
      metrics.force_limit_abort = true;
      metrics.stop_reason = "FORCE_LIMIT_ABORT_HOLD_STOP";
      franka::CartesianVelocities stop{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      return franka::MotionFinished(stop);
    }

    if (!retreat_phase) {
      if (axis_force >= options.max_contact_force_n || force_norm >= options.max_contact_force_n) {
        metrics.contact_threshold_seen = true;
        metrics.stop_reason = "CONTACT_THRESHOLD_RETREAT";
        retreat_phase = true;
        retreat_start_projection = projection;
      } else if (projection >= options.insertion_depth_limit_m) {
        metrics.insertion_depth_reached = true;
        metrics.stop_reason = "INSERTION_DEPTH_RETREAT";
        retreat_phase = true;
        retreat_start_projection = projection;
      } else if (metrics.elapsed_s >= options.max_approach_duration_s) {
        metrics.approach_timeout = true;
        metrics.stop_reason = "APPROACH_DURATION_RETREAT";
        retreat_phase = true;
        retreat_start_projection = projection;
      } else {
        franka::CartesianVelocities output{{axis[0] * options.velocity_m_s,
                                            axis[1] * options.velocity_m_s,
                                            axis[2] * options.velocity_m_s, 0.0, 0.0, 0.0}};
        return output;
      }
    }

    metrics.retreat_executed = true;
    metrics.retreat_distance_m = std::max(0.0, retreat_start_projection - projection);
    if (metrics.retreat_distance_m >= options.retreat_distance_m ||
        metrics.elapsed_s >= options.max_approach_duration_s + options.max_retreat_duration_s) {
      metrics.retreat_completed = metrics.retreat_distance_m >= options.retreat_distance_m;
      if (!metrics.retreat_completed) {
        metrics.stop_reason = "RETREAT_TIMEOUT_HOLD_STOP";
      }
      franka::CartesianVelocities stop{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      return franka::MotionFinished(stop);
    }

    franka::CartesianVelocities retreat{{-axis[0] * options.velocity_m_s,
                                         -axis[1] * options.velocity_m_s,
                                         -axis[2] * options.velocity_m_s, 0.0, 0.0, 0.0}};
    return retreat;
  });
  return metrics;
}

std::string ExecutedJson(const Options& options, const Metrics& metrics) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"ok\": true,\n";
  out << "  \"status\": \"" << kExecutedPendingReviewStatus << "\",\n";
  out << "  \"future_accept_status\": \"" << kFutureAcceptStatus << "\",\n";
  out << "  \"run_id\": \"" << JsonEscape(options.run_id) << "\",\n";
  out << "  \"result_root\": \"" << JsonEscape(options.result_root) << "\",\n";
  out << "  \"runtime_execution_performed\": true,\n";
  out << "  \"fixture_contact_envelope_executed\": " << Bool(metrics.fixture_contact_envelope_executed)
      << ",\n";
  out << "  \"live_micro_motion_executed\": " << Bool(metrics.live_micro_motion_executed) << ",\n";
  out << "  \"fr3_connection\": true,\n";
  out << "  \"fci_state_read\": " << Bool(metrics.fci_state_used) << ",\n";
  out << "  \"motion_command\": " << Bool(metrics.motion_command) << ",\n";
  out << "  \"gripper_command\": false,\n";
  out << "  \"policy_rollout\": false,\n";
  out << "  \"remote_training\": false,\n";
  out << "  \"checkpoint_promotion\": false,\n";
  out << "  \"droid_mutation\": false,\n";
  out << "  \"thread_019e4faf_touched\": false,\n";
  out << "  \"openpi_dependency\": false,\n";
  out << "  \"old_8d_joint_target_action_accepted\": false,\n";
  out << "  \"max_force_norm_n\": " << metrics.max_force_norm_n << ",\n";
  out << "  \"max_axis_force_abs_n\": " << metrics.max_axis_force_abs_n << ",\n";
  out << "  \"max_approach_depth_m\": " << metrics.max_approach_depth_m << ",\n";
  out << "  \"retreat_distance_m\": " << metrics.retreat_distance_m << ",\n";
  out << "  \"retreat_executed\": " << Bool(metrics.retreat_executed) << ",\n";
  out << "  \"retreat_completed\": " << Bool(metrics.retreat_completed) << ",\n";
  out << "  \"force_limit_abort\": " << Bool(metrics.force_limit_abort) << ",\n";
  out << "  \"contact_threshold_seen\": " << Bool(metrics.contact_threshold_seen) << ",\n";
  out << "  \"stop_reason\": \"" << JsonEscape(metrics.stop_reason) << "\",\n";
  out << "  \"postflight_human_final_review_required\": true,\n";
  out << "  \"safe_to_claim_100_percent\": false,\n";
  out << "  \"overall_goal_complete\": false\n";
  out << "}";
  return out.str();
}

void WriteResultFile(const Options& options, const std::string& payload) {
  const std::filesystem::path result_dir =
      std::filesystem::path(options.result_root) / options.run_id;
  std::filesystem::create_directories(result_dir);
  const std::filesystem::path result_path = result_dir / "live_control_body_result.json";
  std::ofstream output(result_path);
  output << payload << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = ParseArgs(argc, argv);
  } catch (const std::exception& exc) {
    Options fallback;
    Reject(fallback, "ARGUMENT_PARSE_ERROR", exc.what());
  }

  if (options.self_check) {
    std::ostringstream extra;
    extra << "  \"live_control_body_source_present\": true,\n";
    extra << "  \"links_libfranka_at_compile_time\": true,\n";
    extra << "  \"uses_cartesian_velocity_control\": true,\n";
    extra << "  \"uses_external_wrench_guard\": true,\n";
    extra << "  \"requires_fixture_axis_base\": true,\n";
    extra << "  \"splits_approach_and_retreat_duration\": true,\n";
    extra << "  \"interactive_prompt\": false\n";
    Emit(CommonJson(options, true, kReadyStatus, extra.str()), 0);
  }

  ValidateApprovalAndGates(options);
  const std::array<double, 3> axis = NormalizedAxisOrReject(options);
  ValidateEnvelope(options);

  if (!options.execute_live) {
    std::ostringstream extra;
    extra << "  \"ready_to_execute_live_control_body\": true,\n";
    extra << "  \"reviewed_live_execution_command_still_required\": true\n";
    Emit(CommonJson(options, true, kReadyNotExecutedStatus, extra.str()), 0);
  }

  try {
    const Metrics metrics = ExecuteLive(options, axis);
    const std::string payload = ExecutedJson(options, metrics);
    WriteResultFile(options, payload);
    Emit(payload, 0);
  } catch (const franka::Exception& exc) {
    Reject(options, "FRANKA_EXCEPTION", exc.what());
  } catch (const std::exception& exc) {
    Reject(options, "LIVE_CONTROL_EXCEPTION", exc.what());
  }
}
