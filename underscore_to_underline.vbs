' Replaces underscores with underline in Word Document 
' without disrupting spacing/formatting as much as reasonably possible

' Author: Timothy Yu
' https://github.com/timothyyu

Sub underscore_to_underline()

    Selection.WholeStory
    ActiveDocument.Range.Select
    Selection.Find.ClearFormatting
    Selection.Find.Replacement.ClearFormatting
    Selection.Find.Replacement.Font.Underline = wdUnderlineSingle
    With Selection.Find
        .Text = "_"
        .Replacement.Text = "^s "
        .Forward = True
        .Wrap = wdFindAsk
        .Format = True
        .MatchCase = False
        .MatchWholeWord = False
        .MatchAllWordForms = False
        .MatchSoundsLike = False
        .MatchWildcards = True
    End With
    Selection.Find.Execute
    Selection.Find.Execute
    Selection.Find.Execute
    Selection.Find.Execute
    Selection.Find.Execute Replace:=wdReplaceAll

End Sub