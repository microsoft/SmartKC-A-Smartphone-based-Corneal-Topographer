package com.example.kt.data.models

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "files")
data class KTFile(
    @PrimaryKey var uri: String,
    var uploaded: Boolean
)