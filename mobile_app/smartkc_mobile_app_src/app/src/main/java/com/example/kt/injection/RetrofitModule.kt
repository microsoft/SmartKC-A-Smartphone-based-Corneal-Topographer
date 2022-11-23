package com.example.kt.injection

import android.content.Context
import com.example.kt.service.FileAPI
import dagger.Module
import dagger.Provides
import dagger.Reusable
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class RetrofitModule {

    @Provides
    @Reusable
    fun provideOkHttp(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.MINUTES)
            .readTimeout(10, TimeUnit.MINUTES)
            .build()
    }

    @Provides
    @Reusable
    fun provideRetrofitInstance(
        okHttpClient: OkHttpClient
    ): Retrofit {
        return Retrofit.Builder().client(okHttpClient).baseUrl("https://olive-readers-visit-139-167-236-150.loca.lt").build()
    }

    @Provides
    @Reusable
    fun provideLanguageAPI(retrofit: Retrofit): FileAPI {
        return retrofit.create(FileAPI::class.java)
    }
}