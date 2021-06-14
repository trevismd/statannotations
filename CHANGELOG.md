## v3
### v3.0
(Explicitly Requires Python 3.6)

 - Refactoring with implementation of `StatTest`
   ([#7](https://github.com/trevismd/statannotations/pull/5)), removing the 
   `stat_func` parameter, and `test_long_name`.
 - Annotations y-positions based on plot coordinates instead of data coordinates 
   ([#5](https://github.com/trevismd/statannotations/pull/5) by [JosephLalli](https://github.com/JosephLalli), fixes https://github.com/webermarcolivier/statannot/issues/21).
 - Add this CHANGELOG

## v2
### v2.8
 - Fix bug on group/box named 0, fixes https://github.com/trevismd/statannotations/issues/10, originally in https://github.com/webermarcolivier/statannot/issues/78. Independently fixed in https://github.com/webermarcolivier/statannot/pull/73 before this issue.