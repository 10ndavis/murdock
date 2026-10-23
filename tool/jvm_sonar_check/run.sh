#!/usr/bin/env bash
# Runs SonarEngine's real per-ping DSP (speaker cancellation, echo clarity, AUTO-TUNE) on a
# desktop JVM with simulated echoes. Needs an Android framework jar for the Context class,
# e.g. Robolectric's android-all:
#   KOTLINC=/path/to/kotlinc ANDROID_ALL=/path/to/android-all.jar tool/jvm_sonar_check/run.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src="$here/../../android/app/src/main/kotlin/com/murdock/murdock"
out=$(mktemp -d)
"${KOTLINC:-kotlinc}" -jvm-target 17 -nowarn -cp "$ANDROID_ALL" -d "$out" \
  "$here/../jvm_startup_check/android/util/Log.kt" "$src/SonarEngine.kt" "$src/SonarConfig.kt" "$here/SonarTuneTest.kt"
# Our Log stub first: the framework one calls native code.
java -cp "$out:$ANDROID_ALL:$(dirname "$(command -v "${KOTLINC:-kotlinc}")")/../lib/kotlin-stdlib.jar" SonarTuneTestKt
