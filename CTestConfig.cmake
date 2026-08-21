# # This file should be placed in the root directory of your project.
# # Then modify the CMakeLists.txt file in the root directory of your
# # project to incorporate the testing dashboard.
# #
# # # The following are required to submit to the CDash dashboard:
# #   ENABLE_TESTING()
# #   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "flash-attention")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")

# Public release note: configure these variables locally if you submit to a CDash instance.
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "")
set(CTEST_DROP_LOCATION "")
set(CTEST_DROP_SITE_CDASH FALSE)

# # Set default timeout to 10 minutes
set(DART_TESTING_TIMEOUT 600)

set(BUILDNAME "${CTEST_PROJECT_NAME}")
set(CTEST_LABELS_FOR_SUBPROJECTS "${BUILDNAME}_LABEL")
