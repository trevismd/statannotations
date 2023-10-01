## v0.6
### v0.6.0
#### Features
- Add option to skip annotation of non-significant results 
  (PR [#95](https://github.com/trevismd/statannotations/pull/95) by 
  [sepro](https://github.com/sepro))

#### Fixes
- Fix keeping annotation with reduced ylim (
  PR [#116](https://github.com/trevismd/statannotations/issues/116) by
  [amkorb](https://github.com/amkorb))
- Fix pvalue legend (usually for NS range)

#### Additional testing and documentation:
  - PR [#84](https://github.com/trevismd/statannotations/pull/84) by
    [JasonMendoza2008 ](https://github.com/JasonMendoza2008)
  - PR [#86](https://github.com/trevismd/statannotations/pull/86) by 
    [mbhall88](https://github.com/mbhall88)
  - PR [#117](https://github.com/trevismd/statannotations/pull/117) by
    [tathey1](https://github.com/tathey1)

## v0.5.0
- Add scipy's Brunner-Munzel test
- Fix applying statannotations for non-string group labels (Issue 
  [#65](https://github.com/trevismd/statannotations/issues/65))
- Get Zenodo DOI

### v0.4.5
- Add MANIFEST.IN (PR [#56](https://github.com/trevismd/statannotations/pull/56)
  by [Matt Chan](https://github.com/thewchan))
- Limit supported Seaborn version to v.0.11.x
- Fix adding annotations with hue if data is passed as arrays 
  (PR [#64](https://github.com/trevismd/statannotations/pull/64) by 
  [getzze](https://github.com/getzze))

### v0.4.4
- The label for Kruskal-Wallis test explicitly states that it is run pairwise 
  (PR [#40](https://github.com/trevismd/statannotations/pull/40) by
  [sepro](https://github.com/sepro))
- Fix broken link in readme
  (PR [#43](https://github.com/trevismd/statannotations/pull/43) by
  [orena1](https://github.com/orena1))
- Fix custom annotations order with respect to the given pairs (Issue 
  [#45](https://github.com/trevismd/statannotations/issues/45))

### v0.4.3
- The `correction_format` parameter allows changing how the multiple 
comparisons correction method  adjusts the annotation when changing a result
to non-significant.
- Fix the `show_test_name` configuration.
- Fix the `verbose` parameter 
 (PR [#37](https://github.com/trevismd/statannotations/pull/37) by 
   [mxposed](https://github.com/mxposed))

### v0.4.2
 - Support of `FacetGrid` with
   - Support empty initialization only defining pairs
   - `plot_and_annotate_facets`

### v0.4.1
 - Support for horizontal orientation

### v0.4.0
 - Major refactoring, change to an Annotator `class` to prepare and add 
   annotations in separate function (now method) calls.
 - Support for `violinplot`
 - Fixes in rendering of non-significant tests results after multiple 
   comparisons correction
 - Fix the printout of the star annotation legend 
 - Fix the failure when providing a dataframe with categories as different 
   columns.
 - Fix a bug where an incorrect xunits calculation resulted in wrong 
   association of points within a box, leading to erroneous max y position for 
   that box.
 - Reduced code complexity, more SOLID.
 - Many unit and integration tests

## v0.3
### v0.3.2
 - Fix `simple` format outputs
 - Fix `ImportError` when applying a multiple comparison correction without 
   statsmodels.
 - Multiple comparison correction is `None` by default, as `statsmodels` is an 
   additional dependency.

### v0.3.1
 - Added support of functions returning more than the two expected values when 
   used in `StatTest`
 - Fix version numbers in CHANGELOG

### v0.3.0
(Explicitly Requires Python 3.6)

 - Refactoring with implementation of `StatTest`
   ([#7](https://github.com/trevismd/statannotations/pull/5)), removing the 
   `stat_func` parameter, and `test_long_name`.
 - Annotations y-positions based on plot coordinates instead of data 
   coordinates 
   (PR [#5](https://github.com/trevismd/statannotations/pull/5) by 
   [JosephLalli](https://github.com/JosephLalli), 
   fixes https://github.com/webermarcolivier/statannot/issues/21).
 - Add this CHANGELOG

## v0.2
### v0.2.8
 - Fix bug on group/box named 0, fixes 
   https://github.com/trevismd/statannotations/issues/10, originally in
   https://github.com/webermarcolivier/statannot/issues/78. Independently 
   fixed in https://github.com/webermarcolivier/statannot/pull/73 before this 
   issue.
