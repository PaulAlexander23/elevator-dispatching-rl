/* A flat C API over elevator::VecEnv, for loading the env as a plain shared
 * library (libelevator_c.so / elevator_c.dll) from ctypes, cffi, C#, Julia...
 * The Python side is elevator_rl/ctypes_env.py.
 *
 * Rules that keep it usable from any language:
 * - Only fixed-width integers, doubles, pointers and one plain struct cross
 *   the boundary: no C++ types, no exceptions, no bool (its size varies).
 * - The library owns an env handle: elevator_vec_create makes it and
 *   elevator_vec_destroy frees it. The caller owns every array.
 * - Functions that can fail return ELEVATOR_OK or ELEVATOR_ERROR (NULL from
 *   elevator_vec_create); elevator_last_error() then says why.
 */
#ifndef ELEVATOR_C_H
#define ELEVATOR_C_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ELEVATOR_C_BUILDING)
#    define ELEVATOR_API __declspec(dllexport)
#  else
#    define ELEVATOR_API __declspec(dllimport)
#  endif
/* The only convention on x64; on 32-bit x86 it is what ctypes.CDLL expects
 * (ctypes.WinDLL would expect __stdcall). */
#  define ELEVATOR_CALL __cdecl
#else
#  define ELEVATOR_API __attribute__((visibility("default")))
#  define ELEVATOR_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever a signature or ElevatorOptions changes. */
#define ELEVATOR_ABI_VERSION 1

enum { ELEVATOR_OK = 0, ELEVATOR_ERROR = 1 };
enum { ELEVATOR_OBS_CUSTOM = 0, ELEVATOR_OBS_BOX = 1, ELEVATOR_OBS_RELATIVE = 2 };
enum { ELEVATOR_ACTION_STEP = 0, ELEVATOR_ACTION_TARGET = 1 };

/* The BuildingEnv constructor arguments other than the config, as
 * elevator::EnvOptions. Fill it with elevator_default_options first. */
typedef struct ElevatorOptions {
  double pickup;
  double empty_serve;
  double progress;
  double waiting;
  double gamma;
  int32_t shaped;            /* 0: reward is passengers delivered only */
  int32_t max_steps;
  int32_t action_mode;       /* ELEVATOR_ACTION_* */
  int32_t observe_direction; /* 0 or 1 */
} ElevatorOptions;

/* Opaque: callers only ever hold a pointer. */
typedef struct ElevatorVecEnv ElevatorVecEnv;

ELEVATOR_API int32_t ELEVATOR_CALL elevator_abi_version(void);
/* sizeof(ElevatorOptions), so a binding can check its own struct layout. */
ELEVATOR_API int32_t ELEVATOR_CALL elevator_options_size(void);
/* Why this thread's last failing call failed. The string stays valid until
 * this thread's next failing call. */
ELEVATOR_API const char* ELEVATOR_CALL elevator_last_error(void);

ELEVATOR_API int32_t ELEVATOR_CALL elevator_default_options(ElevatorOptions* out);

/* `config` is "key=value ..." pairs, as elevator_rl.trace.config_line writes
 * them. Env i's first episode seeds its stream with seed + i. */
ELEVATOR_API ElevatorVecEnv* ELEVATOR_CALL elevator_vec_create(const char* config,
                                                              const ElevatorOptions* options,
                                                              int32_t n_envs, int32_t obs_type,
                                                              uint64_t seed);
/* Accepts NULL. */
ELEVATOR_API void ELEVATOR_CALL elevator_vec_destroy(ElevatorVecEnv* env);

/* These return -1 for a NULL handle. */
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_num_envs(const ElevatorVecEnv* env);
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_n_lifts(const ElevatorVecEnv* env);
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_n_actions(const ElevatorVecEnv* env);
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_observation_size(const ElevatorVecEnv* env);

/* Observation arrays are num_envs x observation_size, of int64 for
 * ELEVATOR_OBS_CUSTOM and float32 otherwise, C-contiguous.
 * `seeds` holds one entry per env (or is NULL); a negative entry keeps that
 * env's seed stream. */
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_reset(ElevatorVecEnv* env, const int64_t* seeds,
                                                     void* obs);

/* `actions` is num_envs x n_lifts, each in [0, n_actions). Finished episodes
 * reset at once: `obs` gets the new first observation, and `terminal_obs`,
 * `episode_returns` and `episode_lengths` the finished episode's; rows of
 * those three for envs that did not finish are left untouched. `rewards`,
 * `dones` and the episode arrays have num_envs entries. On an error nothing
 * is written. */
ELEVATOR_API int32_t ELEVATOR_CALL elevator_vec_step(ElevatorVecEnv* env, const int64_t* actions,
                                                    void* obs, float* rewards, uint8_t* dones,
                                                    void* terminal_obs, float* episode_returns,
                                                    int32_t* episode_lengths);

#ifdef __cplusplus
}
#endif

#endif /* ELEVATOR_C_H */
