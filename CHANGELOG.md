## v0.3
### v0.3.2
 - Fix `simple` format outputs
 - Fix `ImportError` when applying a multiple comparison correction without statsmodels.
 - Multiple comparison correction is `None` by default, as `statsmodels` is an additional dependency.

### v0.3.1
 - Added support of functions returning more than the two expected values when used in `StatTest`
 - Fix version numbers in CHANGELOG

### v0.3.0
(Explicitly Requires Python 3.6)

 - Refactoring with implementation of `StatTest`
   ([#7](https://github.com/trevismd/statannotations/pull/5)), removing the 
   `stat_func` parameter, and `test_long_name`.
 - Annotations y-positions based on plot coordinates instead of data coordinates 
   ([#5](https://github.com/trevismd/statannotations/pull/5) by [JosephLalli](https://github.com/JosephLalli), fixes https://github.com/webermarcolivier/statannot/issues/21).
 - Add this CHANGELOG

## v0.2
### v0.2.8
 - Fix bug on group/box named 0, fixes https://github.com/trevismd/statannotations/issues/10, originally in https://github.com/webermarcolivier/statannot/issues/78. Independently fixed in https://github.com/webermarcolivier/statannot/pull/73 before this issue.
