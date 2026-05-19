"""Tests for feature flag resolution."""

from security.feature_flags import (
    INSIGHT_AGENT_FLAG,
    STRATEGY_AGENT_FLAG,
    has_feature,
    resolve_feature_flags,
)


class TestHasFeature:
    def test_flag_present(self):
        source = {"feature_flags": [INSIGHT_AGENT_FLAG, STRATEGY_AGENT_FLAG]}
        assert has_feature(source, INSIGHT_AGENT_FLAG) is True

    def test_flag_missing(self):
        source = {"feature_flags": [STRATEGY_AGENT_FLAG]}
        assert has_feature(source, INSIGHT_AGENT_FLAG) is False

    def test_none_source(self):
        assert has_feature(None, INSIGHT_AGENT_FLAG) is False

    def test_empty_source(self):
        assert has_feature({}, INSIGHT_AGENT_FLAG) is False

    def test_non_list_flags(self):
        source = {"feature_flags": "not_a_list"}
        assert has_feature(source, INSIGHT_AGENT_FLAG) is False

    def test_non_string_in_list(self):
        source = {"feature_flags": [123, INSIGHT_AGENT_FLAG]}
        assert has_feature(source, INSIGHT_AGENT_FLAG) is True


class TestResolveFeatureFlags:
    def test_from_auth_user(self):
        config = {
            "configurable": {
                "langgraph_auth_user": {
                    "feature_flags": [INSIGHT_AGENT_FLAG]
                }
            }
        }
        flags = resolve_feature_flags(config)
        assert flags == [INSIGHT_AGENT_FLAG]

    def test_from_configurable_fallback(self):
        config = {
            "configurable": {
                "feature_flags": [INSIGHT_AGENT_FLAG, STRATEGY_AGENT_FLAG]
            }
        }
        flags = resolve_feature_flags(config)
        assert INSIGHT_AGENT_FLAG in flags
        assert STRATEGY_AGENT_FLAG in flags

    def test_auth_user_takes_precedence(self):
        config = {
            "configurable": {
                "langgraph_auth_user": {
                    "feature_flags": [INSIGHT_AGENT_FLAG]
                },
                "feature_flags": [STRATEGY_AGENT_FLAG],
            }
        }
        flags = resolve_feature_flags(config)
        assert flags == [INSIGHT_AGENT_FLAG]

    def test_empty_config(self):
        assert resolve_feature_flags({}) == []
        assert resolve_feature_flags(None) == []

    def test_malformed_auth_user_fails_closed(self):
        config = {
            "configurable": {
                "langgraph_auth_user": {
                    "feature_flags": "not_a_list"
                },
                "feature_flags": [STRATEGY_AGENT_FLAG],
            }
        }
        flags = resolve_feature_flags(config)
        assert flags == []
